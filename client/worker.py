import time
import requests
import uuid
import sys
import os
import io
import torch
import threading
import psutil

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Hack to fix imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.protocol import Capability, PeerHeartbeat

# Cache config
os.environ["HF_HOME"] = "H:/gen_ai/llm"

from client.shard_engine import ShardEngine

SERVER_URL = "http://localhost:8002"
DASHBOARD_PORT = 8080  # Local dashboard

# Shared state for dashboard
worker_state = {
    "status": "Initializing",
    "layer_range": "—",
    "processed": 0,
    "ram_percent": 0,
    "vram_percent": 0,
    "logs": []
}

class WorkerClient:
    def __init__(self):
        self.peer_id = str(uuid.uuid4())
        self.capability = Capability(
            vram_mb=8000, 
            compute_score=10.0,
            bandwidth_mbps=100.0,
            peer_id=self.peer_id
        )
        self.registered = False
        self.engine = None
        self.assignment = None
        self.next_peer = None
        self.prev_peer = None 
        
        # Logging
        self.log_file = open(f"worker_{self.peer_id}.log", "w")
        
    def log(self, msg):
        print(msg)
        self.log_file.write(msg + "\n")
        self.log_file.flush()
        # Also push to shared state for dashboard
        worker_state["logs"].append(msg)
        if len(worker_state["logs"]) > 100:
            worker_state["logs"].pop(0)

    def register(self):
        self.log(f"Registering with {SERVER_URL}...")
        try:
            resp = requests.post(f"{SERVER_URL}/register", json=self.capability.dict())
            if resp.status_code == 200:
                data = resp.json()
                self.peer_id = data["peer_id"]
                self.registered = True
                self.log(f"Registered as {self.peer_id}")
            else:
                self.log(f"Registration failed: {resp.text}")
        except Exception as e:
            self.log(f"Connection failed: {e}")

    def loop(self):
        if not self.registered:
            self.register()
        
        while True:
            if not self.registered:
                time.sleep(5)
                self.register()
                continue
                
            try:
                # 1. Heartbeat
                hb = PeerHeartbeat(
                    peer_id=self.peer_id,
                    capabilities=self.capability,
                    status="idle" if not self.engine else "ready"
                )
                resp = requests.post(f"{SERVER_URL}/heartbeat", json=hb.dict())
                data = resp.json()
                action = data.get("action")
                
                # 2. Handle Action
                if action == "assign_shard":
                    new_assignment = data.get("assignment")
                    cluster_map = data.get("cluster_map")
                    if self.assignment != new_assignment:
                        self.log(f"Received Assignment: {new_assignment}")
                        self.init_shard(new_assignment, cluster_map)
                        
                elif action == "wait":
                    if not self.engine:
                        # self.log("Waiting for assignment...") 
                        # Don't spam log
                        time.sleep(2)
                        continue
                
                # 3. Poll for Data (if engine ready)
                if self.engine:
                    self.poll_and_process()
                    
                time.sleep(0.1) # Fast poll
            except Exception as e:
                self.log(f"Error in loop: {e}")
                time.sleep(5)

    def init_shard(self, assignment: dict, cluster_map: dict = None):
        self.assignment = assignment
        
        # Determine topology from cluster_map
        if cluster_map:
            my_start = assignment["start_layer"]
            my_end = assignment["end_layer"]
            
            for pid, shard in cluster_map.items():
                # Find Next
                if shard["start_layer"] == my_end:
                    self.next_peer = pid
                    self.log(f"Topology: Next peer is {pid}")
                
                # Find Prev
                if shard["end_layer"] == my_start:
                    self.prev_peer = pid
                    self.log(f"Topology: Prev peer is {pid}")
        
        if not self.engine:
            self.log("Initializing Engine...")
            self.engine = ShardEngine(
                model_name=assignment["model_id"],
                start_layer=assignment["start_layer"],
                end_layer=assignment["end_layer"],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            self.log("Engine Ready!")
            
            # Update dashboard state
            worker_state["status"] = "Ready"
            worker_state["layer_range"] = f"L{assignment['start_layer']}-{assignment['end_layer']}"

    def poll_and_process(self):
        # Poll server relay
        try:
            resp = requests.get(f"{SERVER_URL}/relay/poll/{self.peer_id}")
            if resp.status_code == 200:
                content = resp.content
                if not content: 
                     return
                     
                # Read Metadata
                msg_type = resp.headers.get("X-Msg-Type", "activation")
                req_id = resp.headers.get("X-Req-Id", "default")
                
                self.log(f"Received {msg_type} ({len(content)} bytes)")
                
                # Deserialize
                try:
                    buffer = io.BytesIO(content)
                    tensor = torch.load(buffer, map_location=self.engine.device, weights_only=False)
                    self.log(f"Deserialized tensor: {tensor.shape}, dtype={tensor.dtype}")
                except Exception as e:
                    self.log(f"Deserialization Failed: {e}")
                    return

                if msg_type == "activation":
                    # FORWARD PASS
                    self.log("Starting forward_shard...")
                    try:
                        output = self.engine.forward_shard(tensor)
                        self.log(f"Forward complete. Output: {output.shape}")
                        worker_state["processed"] += 1
                    except Exception as e:
                        self.log(f"Forward Failed: {e}")
                        return
                    
                    # Send to next peer
                    if self.next_peer:
                        out_buffer = io.BytesIO()
                        torch.save(output.cpu(), out_buffer)
                        out_data = out_buffer.getvalue()
                        
                        requests.post(
                            f"{SERVER_URL}/relay/send/{self.next_peer}", 
                            params={"msg_type": "activation", "req_id": req_id},
                            data=out_data
                        )
                    else:
                        self.log("Sink reached. Computing Loss (Mock)...")
                        
                        # Send result to verification sink (for test_network.py)
                        out_buffer = io.BytesIO()
                        torch.save(output.cpu(), out_buffer)
                        out_data = out_buffer.getvalue()
                        requests.post(f"{SERVER_URL}/relay/send/result_sink", data=out_data)

                        # Mock Loss: Minimize L2 Norm (Should go to 0)
                        loss = output.pow(2).mean() 
                        self.log(f"Loss: {loss.item()}")
                        
                        # Report Loss
                        try:
                            requests.post(f"{SERVER_URL}/relay/send/loss_sink", data=str(loss.item()).encode())
                        except:
                            pass
                        
                        loss.backward()
                        
                        input_grad = self.engine.last_input.grad
                        if input_grad is not None:
                            # Send upstream
                            self.send_gradient_upstream(input_grad.cpu(), req_id)
                        else:
                             self.log("Sink: No input gradient to send upstream.")

                        # Optimization Step
                        self.engine.step()
                        self.log("Optimizer Step executed (Sink).")

                elif msg_type == "gradient":
                    # BACKWARD PASS
                    input_grad = self.engine.backward_shard(tensor)
                    
                    if input_grad is not None:
                         self.send_gradient_upstream(input_grad, req_id)
                    else:
                         self.log("Source reached. Gradients at input (or None). Step optimizer?")
                         # Signal completion for verification script
                         requests.post(f"{SERVER_URL}/relay/send/backward_sink", data=b"BACKWARD_DONE")
                         pass
                    
                    # Optimization Step
                    self.engine.step()
                    self.log("Optimizer Step executed (Backward Node).")

        except Exception as e:
            # self.log(f"Poll error: {e}")
            pass

    def send_gradient_upstream(self, gradient: torch.Tensor, req_id: str):
        if self.prev_peer:
             # self.log(f"Sending gradient to {self.prev_peer}")
             out_buffer = io.BytesIO()
             torch.save(gradient, out_buffer)
             out_data = out_buffer.getvalue()
             
             requests.post(
                f"{SERVER_URL}/relay/send/{self.prev_peer}",
                params={"msg_type": "gradient", "req_id": req_id},
                data=out_data
             )
        else:
             self.log("No previous peer (Source). Backward pass complete.")

# ============ Local Dashboard Server ============
dashboard_app = FastAPI(title="Worker Agent Dashboard")

static_path = os.path.join(os.path.dirname(__file__), 'static')
dashboard_app.mount("/static", StaticFiles(directory=static_path, html=True), name="static")

@dashboard_app.get("/")
def dashboard_root():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_path, "index.html"))

@dashboard_app.get("/state")
def get_state():
    # Update resource usage
    worker_state["ram_percent"] = psutil.virtual_memory().percent
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) == 2:
                used, total = float(parts[0]), float(parts[1])
                worker_state["vram_percent"] = (used / total) * 100
    except:
        pass
    
    return JSONResponse(worker_state)

@dashboard_app.get("/logs")
def get_logs():
    return JSONResponse({"logs": worker_state["logs"][-50:]})

def run_dashboard():
    uvicorn.run(dashboard_app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning")

if __name__ == "__main__":
    # Start dashboard in background thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()
    print(f"Agent Dashboard running at http://localhost:{DASHBOARD_PORT}")
    
    client = WorkerClient()
    client.loop()
