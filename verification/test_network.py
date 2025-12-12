import subprocess
import time
import requests
import torch
import io
import sys
import os

# Env 
os.environ["HF_HOME"] = "H:/gen_ai/llm"

SERVER_URL = "http://localhost:8002"

def run_test():
    print("WARNING: Assuming Server is running on 8002.")
    # print("Starting Coordinator...")
    # server_proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.coordinator:app", "--port", "8001"], cwd="g:\\python\\hdgAtHome")
    
    # time.sleep(5) # Wait for startup
    
    workers = []
    print("Starting Worker 1...")
    workers.append(subprocess.Popen([sys.executable, "-u", "client/worker.py"], cwd="g:\\python\\hdgAtHome"))
    
    print("Starting Worker 2...")
    workers.append(subprocess.Popen([sys.executable, "-u", "client/worker.py"], cwd="g:\\python\\hdgAtHome"))
    
    print("Waiting for registration and assignments...")
    time.sleep(10)
    
    # Check status
    try:
        resp = requests.get(f"{SERVER_URL}/")
        print(f"Server Status: {resp.json()}")
        
        # We need to find the Head Peer (Start Layer 0)
        head_peer = None
        assignments = resp.json().get("assignments", {})
        
        while len(assignments) < 2:
            print("Waiting for assignments...", end='\r')
            time.sleep(2)
            resp = requests.get(f"{SERVER_URL}/")
            assignments = resp.json().get("assignments", {})
            
        print(f"\nAssignments: {assignments.keys()}")
            
        for pid, shard in assignments.items():
            if shard["start_layer"] == 0:
                head_peer = pid
                break
                
        if not head_peer:
            print("FATAL: No head peer found.")
            return

        print(f"Head Peer: {head_peer}")
        
        # Inject Input IDs
        print("Injecting Input IDs to Head Peer...")
        input_ids = torch.randint(0, 50257, (1, 10), dtype=torch.long)
        
        # Serialize
        buf = io.BytesIO()
        torch.save(input_ids, buf)
        data = buf.getvalue()
        
        requests.post(f"{SERVER_URL}/relay/send/{head_peer}", data=data)
        
        # Poll Result Sink
        print("Polling Result Sink...")
        for i in range(20):
            res = requests.get(f"{SERVER_URL}/relay/poll/result_sink")
            if res.status_code == 200 and res.content:
                print("Received Final Result!")
                # Deserialize
                final_buf = io.BytesIO(res.content)
                final_tensor = torch.load(final_buf, weights_only=False)
                print(f"Final Tensor Shape: {final_tensor.shape}")
                
                # Check shape (1, 10, 50257)
                if final_tensor.shape == (1, 10, 50257):
                    print("SUCCESS: Network Verification Passed!")
                else:
                    print(f"FAILURE: Unexpected shape {final_tensor.shape}")
                
            time.sleep(1)
            
        # Poll Backward Sink
        print("Polling Backward Sink (Source Reached)...")
        for i in range(20):
            res = requests.get(f"{SERVER_URL}/relay/poll/backward_sink")
            if res.status_code == 200 and res.content:
                if res.content == b"BACKWARD_DONE":
                     print("SUCCESS: Backward Pass Verified (Source Reached)!")
                     break
            time.sleep(1)
        else:
            print("Timeout waiting for Backward validation.")
            
    except Exception as e:
        print(f"Failed to contact server: {e}")
        
    print("Stopping Cluster...")
    for p in workers:
        p.terminate()
    # server_proc.terminate()

if __name__ == "__main__":
    run_test()
