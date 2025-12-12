import torch
from transformers import AutoConfig, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
config = AutoConfig.from_pretrained(model_name)

print(f"Config: hidden_size={config.hidden_size}, num_attention_heads={config.num_attention_heads}")
print(f"Calc head_dim = {config.hidden_size / config.num_attention_heads}")

# Initialize model on CPU (cheap for config only/empty weights)
model = AutoModelForCausalLM.from_config(config)

rotary = model.model.rotary_emb
print(f"Rotary Class: {type(rotary)}")
print(f"Rotary Dim (from attribute if exists): {getattr(rotary, 'dim', 'N/A')}")

# Create dummy inputs
batch = 1
seq = 10
device = "cpu"
# Real tensors for rotary forward
position_ids = torch.arange(0, seq, dtype=torch.long, device=device).unsqueeze(0)
dummy_states = torch.randn(batch, seq, config.hidden_size, device=device)

# Move rotary to cpu for test
import copy
rotary_cpu = copy.deepcopy(rotary).to(device)

try:
    cos, sin = rotary_cpu(dummy_states, position_ids)
    print(f"Cos Shape raw: {cos.shape}")
    print(f"Sin Shape raw: {sin.shape}")
except Exception as e:
    print(f"Rotary forward failed: {e}")

