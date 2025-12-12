import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os

class ShardEngine(nn.Module):
    def __init__(self, model_name: str, start_layer: int, end_layer: int, device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        
        print(f"Initializing ShardEngine for {model_name} layers [{start_layer}-{end_layer}] on {device}")
        self.config = AutoConfig.from_pretrained(model_name)
        
        # We need to construct a partial model.
        # This is hacky. We will load the full model structure on meta device, 
        # then only materialize the layers we own.
        
        with init_empty_weights():
            # We assume a Llama-like structure: model.layers
            self.full_model = AutoModelForCausalLM.from_config(self.config)
            
        # Identify the layers we need to keep
        if "gpt2" in model_name.lower():
             # GPT2 structure: transformer.h
             if not hasattr(self.full_model, "transformer") or not hasattr(self.full_model.transformer, "h"):
                  raise ValueError(f"Model {model_name} architecture not supported (cannot find .transformer.h)")
             self.num_layers = len(self.full_model.transformer.h)
        else:
             # Llama structure
            if not hasattr(self.full_model, "model") or not hasattr(self.full_model.model, "layers"):
                 raise ValueError(f"Model {model_name} architecture not supported (cannot find .model.layers)")
            self.num_layers = self.config.num_hidden_layers

        # We need to manually construct a ModuleList of just our layers
        device_map = {}
        
        is_gpt2 = "gpt2" in model_name.lower()
        
        if is_gpt2:
             # Map embeddings
            if start_layer == 0:
                device_map["transformer.wte"] = device
                device_map["transformer.wpe"] = device
                device_map["transformer.drop"] = device # Dropout
            else:
                device_map["transformer.wte"] = "meta"
                device_map["transformer.wpe"] = "meta"
                device_map["transformer.drop"] = "meta"
            
            # Map layers
            for i in range(self.num_layers):
                if start_layer <= i < end_layer:
                    device_map[f"transformer.h.{i}"] = device
                else:
                    device_map[f"transformer.h.{i}"] = "meta"
                    
            # Map final norm and head
            if end_layer >= self.num_layers:
                device_map["transformer.ln_f"] = device
                device_map["lm_head"] = device 
            else:
                 device_map["transformer.ln_f"] = "meta"
                 device_map["lm_head"] = "meta"

        else:
            # Llama Mapping (Existing logic)
            # Map embeddings
            if start_layer == 0:
                device_map["model.embed_tokens"] = device
            else:
                device_map["model.embed_tokens"] = "meta"
                
            # Map layers
            for i in range(self.config.num_hidden_layers):
                if start_layer <= i < end_layer:
                    device_map[f"model.layers.{i}"] = device
                else:
                    device_map[f"model.layers.{i}"] = "meta"
                    
            # Map final norm and head
            if end_layer >= self.config.num_hidden_layers:
                device_map["model.norm"] = device
                device_map["lm_head"] = device
            else:
                device_map["model.norm"] = "meta"
                device_map["lm_head"] = "meta"
            
        print(f"Device Map generated. Materializing only assigned layers...")
        
        # Determine module root for iteration
        
        # Materialize Logic (Generic)
        # Assuming manual materialization via recursion in previous step is enough if we call it correctly.
        # But we need to call _materialize_module on specific parts.
        
        if is_gpt2:
            if start_layer == 0:
                self._materialize_module(self.full_model.transformer.wte)
                self._materialize_module(self.full_model.transformer.wpe)
            
            for i in range(start_layer, end_layer):
                self._materialize_module(self.full_model.transformer.h[i])
                
            if end_layer >= self.num_layers:
                self._materialize_module(self.full_model.transformer.ln_f)
                self._materialize_module(self.full_model.lm_head)
                
        else:
            # Materialize Embeddings if needed
            if start_layer == 0:
                self._materialize_module(self.full_model.model.embed_tokens)
                
            # Materialize Layers
            for i in range(start_layer, end_layer):
                self._materialize_module(self.full_model.model.layers[i])
                
            # Materialize Head if needed
            if end_layer >= self.config.num_hidden_layers:
                self._materialize_module(self.full_model.model.norm)
                self._materialize_module(self.full_model.lm_head)
            
        # Materialize Rotary Embedding (Must be done!)
        if "llama" in model_name.lower():
             if hasattr(self.full_model.model, "rotary_emb"):
                self._materialize_module(self.full_model.model.rotary_emb)
                self.full_model.model.rotary_emb.to(self.device)
                
        self.model = self.full_model
        
        # Optimizer
        # Only optimize parameters that are on this device
        params = [p for p in self.model.parameters() if p.device.type != 'meta']
        self.optimizer = torch.optim.AdamW(params, lr=1e-4)
        
    def _materialize_module(self, module):
        """Recursively move a meta module to real device with random init (for now)"""
        for name, child in module.named_children():
            self._materialize_module(child)
            
        for name, param in module.named_parameters(recurse=False):
            if param.device.type == 'meta':
                # Create a real parameter
                # Use normal init, strict matching of shape
                new_param = torch.nn.Parameter(torch.empty_like(param, device=self.device).normal_())
                setattr(module, name, new_param)
        
        for name, buf in module.named_buffers(recurse=False):
            if buf.device.type == 'meta':
                new_buf = torch.empty_like(buf, device=self.device).normal_()
                setattr(module, name, new_buf)

    def forward_shard(self, hidden_states: torch.Tensor):
        # Cache input for backward
        # Note: If hidden_states comes from network (detached), we must set requires_grad=True
        # to allow autograd to compute gradients with respect to it.
        if hidden_states.dtype != torch.long and not hidden_states.requires_grad:
            hidden_states = hidden_states.detach().requires_grad_(True)
            
        self.last_input = hidden_states
        
        is_gpt2 = "gpt2" in self.model_name.lower()
        
        # 1. Embeddings (if first shard)
        if self.start_layer == 0:
             # Special case: If input is token IDs (long), we cannot set requires_grad directly.
             # We let the embedding layer handle it.
             if hidden_states.dtype == torch.long:
                 # Reset requires_grad for IDs as it errors out
                 hidden_states.requires_grad = False
                 self.last_input = hidden_states # Save IDs used
                 
                 if is_gpt2:
                     hidden_states = self.model.transformer.wte(hidden_states)
                     seq_len = hidden_states.size(1)
                     pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
                     pos_embed = self.model.transformer.wpe(pos_ids)
                     hidden_states = hidden_states + pos_embed
                 else:
                    hidden_states = self.model.model.embed_tokens(hidden_states)

        batch_size, seq_length = hidden_states.shape[:2]
        past_key_values_length = 0 
        device = hidden_states.device
        
        # Rotary / Position stuff
        position_ids = None
        cos, sin = None, None
        
        if not is_gpt2:
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            
            if hasattr(self.model.model, "rotary_emb"):
                rotary_emb = self.model.model.rotary_emb
                cos, sin = rotary_emb(hidden_states, position_ids)

        # 2. Run Layers
        for i in range(self.start_layer, self.end_layer):
            if is_gpt2:
                layer = self.model.transformer.h[i]
                layer_out = layer(hidden_states)[0]
            else:
                layer = self.model.model.layers[i]
                layer_out = layer(
                     hidden_states,
                     attention_mask=None,
                     position_ids=position_ids,
                     position_embeddings=(cos, sin)
                )[0]
                
            hidden_states = layer_out
            
        # 3. If last layer, run head
        if self.end_layer >= self.config.num_hidden_layers:
            if is_gpt2:
                hidden_states = self.model.transformer.ln_f(hidden_states)
                logits = self.model.lm_head(hidden_states)
            else:
                hidden_states = self.model.model.norm(hidden_states)
                logits = self.model.lm_head(hidden_states)
            
            self.last_output = logits
            return logits
            
        self.last_output = hidden_states
        return hidden_states

    def backward_shard(self, grad_output: torch.Tensor):
        """
        Run backward pass using the cached graph from forward_shard.
        grad_output: Gradients flowing from the Next Peer (or Loss).
        Returns: Gradients to send to Previous Peer.
        """
        if not hasattr(self, 'last_output') or self.last_output is None:
            raise RuntimeError("Cannot run backward: No forward pass cached.")
            
        # Run backward
        # .backward() computes grad for leaf nodes (weights) + inputs (if requires_grad=True)
        # We need to retain_graph if we plan to step multiple times? No, usually once per batch.
        
        # Ensure grad_output is on correct device
        grad_output = grad_output.to(self.device)
        
        # self.last_output is the root of the graph we built in forward_shard
        self.last_output.backward(grad_output)
        
        # Now, we want the gradient with respect to self.last_input
        # This is what we send to previous peer.
        
        if self.start_layer == 0:
            # First shard: No previous peer for activation gradients.
            # But we might have computed gradients for embeddings (weights).
            # Return None or empty.
            return None
        
        if self.last_input.grad is None:
            print("WARNING: last_input.grad is None! Did you detach it incorrectly?")
            return None
            
        return self.last_input.grad.cpu()

    def step(self):
        """
        Perform optimization step.
        """
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()
