
import requests
import time
import torch
import io
import sys

SERVER_URL = "http://localhost:8002"

def get_head_peer():
    # Poll root to find assignment layer 0
    resp = requests.get(f"{SERVER_URL}/")
    data = resp.json()
    assignments = data["assignments"]
    for pid, assign in assignments.items():
        if assign["start_layer"] == 0:
            return pid
    return None

def train_loop():
    print("Waiting for cluster to stabilize...")
    head_peer = None
    while not head_peer:
        head_peer = get_head_peer()
        if not head_peer:
            time.sleep(1)
            print("Waiting for head peer...", end='\r')
            
    print(f"\nHead Peer found: {head_peer}")
    print("Starting Training Loop...")
    
    # Dummy Dataset
    vocab_size = 50257
    seq_len = 10
    
    # OVERFITTING TEST: Fixed Batch
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
    buffer = io.BytesIO()
    torch.save(input_ids, buffer)
    fixed_data = buffer.getvalue()
    
    step = 0
    while True:
        # 1. Send Fixed Batch
        req_id = f"step_{step}"
        print(f"Dispatching Step {step}...", end=' ')
        requests.post(
            f"{SERVER_URL}/relay/send/{head_peer}",
            params={"msg_type": "activation", "req_id": req_id},
            data=fixed_data
        )
        
        # 2. Wait for Backward Completion AND Loss
        done = False
        start_wait = time.time()
        loss_val = None
        
        while not done:
            # Poll completion
            res = requests.get(f"{SERVER_URL}/relay/poll/backward_sink")
            if res.status_code == 200 and res.content == b"BACKWARD_DONE":
                done = True
            
            # Poll loss
            res_loss = requests.get(f"{SERVER_URL}/relay/poll/loss_sink")
            if res_loss.status_code == 200 and res_loss.content:
                 try:
                     loss_val = float(res_loss.content)
                 except:
                     pass
            
            if not done:
                time.sleep(0.1)
                
            if time.time() - start_wait > 30:
                 print("\nTimeout waiting for step completion!")
                 break
        
        if done:
            print(f"Done. Loss: {loss_val}")
            step += 1
            time.sleep(0.1)
        else:
            print("Retrying or aborting...")
            time.sleep(2)

if __name__ == "__main__":
    train_loop()
