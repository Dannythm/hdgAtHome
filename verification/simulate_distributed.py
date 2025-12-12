import sys
import os
import torch
from transformers import AutoConfig

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure Cache
os.environ["HF_HOME"] = "H:/gen_ai/llm"

from client.shard_engine import ShardEngine
from server.partitioner import Partitioner

def run_simulation():
    print("Starting Distributed Simulation...")
    
    # Use a small model for testing
    model_name = "H:/gen_ai/llm/gpt2"
    # model_name = "H:/gen_ai/llm/TinyLlama-1.1B-Chat-v1.0"
    
    # 1. Setup Server/Partitioner
    print("1. setup Partitioner")
    config = AutoConfig.from_pretrained(model_name)
    partitioner = Partitioner({
        "num_hidden_layers": config.num_hidden_layers,
        "model_id": model_name
    })
    
    # 2. Assign Shards
    print(f"Model has {config.num_hidden_layers} layers.")
    peers = [
        {"peer_id": "worker_1", "capabilities": {}},
        {"peer_id": "worker_2", "capabilities": {}}
    ]
    assignments = partitioner.partition(peers)
    print("Assignments:", assignments)
    
    # 3. Instantiate Workers (Shards)
    # We use 'cpu' to avoid OOM or CUDA complexity for this quick test unless user really wants CUDA.
    # User said "I have 2 gpus", so let's try to use them if available, else CPU.
    
    device_1 = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
    device_2 = "cuda:1" if torch.cuda.device_count() > 1 else "cpu" # Fallback to CPU if only 1 GPU
    
    if device_2 == "cpu" and device_1.startswith("cuda"):
        # If we have 1 GPU, maybe split between GPU and CPU? 
        # Or just put both on GPU? Pipelining on same GPU is valid for logic test.
        print("Only 1 GPU detect (or 0), putting Shard 2 on same device as Shard 1 or CPU.")
        device_2 = device_1 

    print(f"Worker 1 on {device_1}")
    print(f"Worker 2 on {device_2}")

    shard1_def = assignments["worker_1"]
    shard2_def = assignments["worker_2"]
    
    print("Initializing Shard 1...")
    engine1 = ShardEngine(
        model_name, 
        shard1_def.start_layer, 
        shard1_def.end_layer, 
        device=device_1
    )
    
    print("Initializing Shard 2...")
    engine2 = ShardEngine(
        model_name, 
        shard2_def.start_layer, 
        shard2_def.end_layer, 
        device=device_2
    )
    
    # 4. Run Forward Pass
    print("\n--- Running Forward Pass ---")
    
    # Create dummy input (Batch size 1, seq len 10)
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device_1)
    
    # Step 1: Worker 1
    # Expects input_ids because it's the first shard
    activations_1 = engine1.forward_shard(input_ids)
    print(f"Shard 1 Output Shape: {activations_1.shape} on {activations_1.device}")
    
    # Simulate Network Transfer (Move to device 2)
    activations_1_transferred = activations_1.to(device_2).detach().requires_grad_(True)
    
    # Step 2: Worker 2
    # Expects hidden_states
    logits = engine2.forward_shard(activations_1_transferred)
    print(f"Shard 2 Output (Logits) Shape: {logits.shape} on {logits.device}")
    
    # 5. Run Backward Pass
    print("\n--- Running Backward Pass ---")
    target = torch.randint(0, config.vocab_size, (1, 10)).to(device_2)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    # Logits: [1, 10, vocab_size], Target: [1, 10]
    # Flatten for loss
    loss = loss_fn(logits.view(-1, config.vocab_size), target.view(-1))
    print(f"Loss: {loss.item()}")
    
    loss.backward()
    
    print("Backward pass through Shard 2 complete.")
    print(f"Input Gradients at Shard 2 boundary: {activations_1_transferred.grad.shape if activations_1_transferred.grad is not None else 'None'}")
    
    if activations_1_transferred.grad is not None:
        # Simulate Network Transfer of Gradients (Back to device 1)
        grad_back = activations_1_transferred.grad.to(device_1)
        
        # Backward pass through Shard 1
        # We need to backward from the endpoint of Shard 1 using the gradients from Shard 2
        activations_1.backward(grad_back)
        print("Backward pass through Shard 1 complete.")
        
        # Check if some weights have grad
        # engine1.model.model.embed_tokens.weight.grad
        if "gpt2" in model_name:
             embed_grad = engine1.model.transformer.wte.weight.grad
        else:
             embed_grad = engine1.model.model.embed_tokens.weight.grad
        if embed_grad is not None:
            print(f"Embeddings Gradient Norm: {embed_grad.norm().item()}")
            print("SUCCESS: Gradient flowed all the way back to input!")
        else:
            print("FAILURE: No gradient on embeddings.")
    else:
        print("FAILURE: No gradient w.r.t input of Shard 2.")

if __name__ == "__main__":
    run_simulation()
