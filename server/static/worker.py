#!/usr/bin/env python3
"""
hdg@home Worker - Distributed AI Training Contributor

Run this script to contribute your GPU power to distributed model training.

Usage:
    python worker.py

Requirements:
    pip install torch transformers accelerate requests fastapi uvicorn psutil

Configuration:
    Set COORDINATOR_URL below to your coordinator's address.
"""

import time
import requests
import uuid
import sys
import os
import io
import torch
import threading
import psutil

# ============ CONFIGURATION ============
COORDINATOR_URL = "http://localhost:8002"  # Change this to the public coordinator
NICKNAME = None  # Optional: Set your contributor nickname
# =======================================

try:
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import JSONResponse, HTMLResponse
    import uvicorn
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install torch transformers accelerate requests fastapi uvicorn psutil")
    sys.exit(1)

# Check for transformers
try:
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    print("Missing transformers. Install with:")
    print("  pip install transformers accelerate")
    sys.exit(1)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ██╗  ██╗██████╗  ██████╗  █████╗ ██╗  ██╗ ██████╗ ███╗   ███╗███████╗   ║
║     ██║  ██║██╔══██╗██╔════╝ ██╔══██╗██║  ██║██╔═══██╗████╗ ████║██╔════╝   ║
║     ███████║██║  ██║██║  ███╗███████║███████║██║   ██║██╔████╔██║█████╗     ║
║     ██╔══██║██║  ██║██║   ██║██╔══██║██╔══██║██║   ██║██║╚██╔╝██║██╔══╝     ║
║     ██║  ██║██████╔╝╚██████╔╝██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║███████╗   ║
║     ╚═╝  ╚═╝╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ║
║                                                              ║
║     Distributed AI Training - Thank you for contributing!   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print(f"Coordinator: {COORDINATOR_URL}")
print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# ============ Protocol Models ============
from pydantic import BaseModel
from typing import Optional, List

class Capability(BaseModel):
    peer_id: Optional[str] = None
    vram_mb: int = 0
    compute_score: float = 1.0
    bandwidth_mbps: float = 100.0

class PeerHeartbeat(BaseModel):
    peer_id: str
    capabilities: Capability
    status: str = "idle"
    timestamp: float = 0.0

# ============ Shard Engine ============
class ShardEngine:
    def __init__(self, model_name: str, start_layer: int, end_layer: int, device: str = "cuda"):
        self.model_name = model_name
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.device = device
        self.model = None
        self.config = None
        self.last_input = None
        self.last_output = None
        self.optimizer = None
        
        print(f"  Loading model config: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name)
        
        print(f"  Initializing layers {start_layer}-{end_layer} on {device}")
        
        # Load full model with meta device for unneeded layers
        device_map = {}
        is_gpt2 = "gpt2" in model_name.lower()
        
        for i in range(self.config.num_hidden_layers):
            layer_key = f"transformer.h.{i}" if is_gpt2 else f"model.layers.{i}"
            if start_layer <= i < end_layer:
                device_map[layer_key] = device
            else:
                device_map[layer_key] = "meta"
        
        # Embeddings and head
        if is_gpt2:
            device_map["transformer.wte"] = device if start_layer == 0 else "meta"
            device_map["transformer.wpe"] = device if start_layer == 0 else "meta"
            device_map["transformer.ln_f"] = device if end_layer >= self.config.num_hidden_layers else "meta"
            device_map["lm_head"] = device if end_layer >= self.config.num_hidden_layers else "meta"
        else:
            device_map["model.embed_tokens"] = device if start_layer == 0 else "meta"
            device_map["model.norm"] = device if end_layer >= self.config.num_hidden_layers else "meta"
            device_map["lm_head"] = device if end_layer >= self.config.num_hidden_layers else "meta"
        
        self.full_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float32
        )
        self.model = self.full_model
        
        # Optimizer
        params = [p for p in self.model.parameters() if p.device.type != 'meta']
        self.optimizer = torch.optim.AdamW(params, lr=1e-4)
        
        print(f"  ✓ Engine ready!")

    def forward_shard(self, hidden_states: torch.Tensor):
        if hidden_states.dtype != torch.long and not hidden_states.requires_grad:
            hidden_states = hidden_states.detach().requires_grad_(True)
        self.last_input = hidden_states
        
        is_gpt2 = "gpt2" in self.model_name.lower()
        
        if self.start_layer == 0 and hidden_states.dtype == torch.long:
            if is_gpt2:
                hidden_states = self.model.transformer.wte(hidden_states)
                seq_len = hidden_states.size(1)
                pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
                hidden_states = hidden_states + self.model.transformer.wpe(pos_ids)
            else:
                hidden_states = self.model.model.embed_tokens(hidden_states)
        
        for i in range(self.start_layer, self.end_layer):
            if is_gpt2:
                hidden_states = self.model.transformer.h[i](hidden_states)[0]
            else:
                hidden_states = self.model.model.layers[i](hidden_states)[0]
        
        if self.end_layer >= self.config.num_hidden_layers:
            if is_gpt2:
                hidden_states = self.model.transformer.ln_f(hidden_states)
            else:
                hidden_states = self.model.model.norm(hidden_states)
            hidden_states = self.model.lm_head(hidden_states)
        
        self.last_output = hidden_states
        return hidden_states

    def backward_shard(self, output_grad: torch.Tensor):
        if self.last_output is None:
            return None
        self.last_output.backward(output_grad, retain_graph=True)
        if self.last_input is not None and self.last_input.grad is not None:
            return self.last_input.grad.cpu()
        return None

    def step(self):
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()

# ============ Shared State ============
worker_state = {
    "status": "Initializing",
    "layer_range": "—",
    "processed": 0,
    "ram_percent": 0,
    "vram_percent": 0,
    "logs": []
}

# ============ Worker Client ============
class WorkerClient:
    def __init__(self):
        self.peer_id = str(uuid.uuid4())
        self.capability = Capability(
            vram_mb=int(torch.cuda.get_device_properties(0).total_memory / 1024**2) if torch.cuda.is_available() else 0,
            compute_score=10.0,
            bandwidth_mbps=100.0,
            peer_id=self.peer_id
        )
        self.registered = False
        self.engine = None
        self.assignment = None
        self.next_peer = None
        self.prev_peer = None

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        worker_state["logs"].append(line)
        if len(worker_state["logs"]) > 100:
            worker_state["logs"].pop(0)

    def register(self):
        self.log(f"Connecting to {COORDINATOR_URL}...")
        try:
            resp = requests.post(f"{COORDINATOR_URL}/register", json=self.capability.dict())
            if resp.status_code == 200:
                self.peer_id = resp.json()["peer_id"]
                self.registered = True
                self.log(f"✓ Registered as {self.peer_id[:8]}...")
                worker_state["status"] = "Registered"
            else:
                self.log(f"✗ Registration failed: {resp.text}")
        except Exception as e:
            self.log(f"✗ Connection failed: {e}")

    def loop(self):
        if not self.registered:
            self.register()
        
        while True:
            if not self.registered:
                time.sleep(5)
                self.register()
                continue
            
            try:
                hb = PeerHeartbeat(
                    peer_id=self.peer_id,
                    capabilities=self.capability,
                    status="idle" if not self.engine else "ready"
                )
                resp = requests.post(f"{COORDINATOR_URL}/heartbeat", json=hb.dict())
                data = resp.json()
                action = data.get("action")
                
                if action == "assign_shard":
                    assignment = data.get("assignment")
                    cluster_map = data.get("cluster_map")
                    if self.assignment != assignment:
                        self.log(f"Received assignment: layers {assignment['start_layer']}-{assignment['end_layer']}")
                        self.init_shard(assignment, cluster_map)
                
                if self.engine:
                    self.poll_and_process()
                
                time.sleep(0.1)
            except Exception as e:
                self.log(f"Error: {e}")
                time.sleep(5)

    def init_shard(self, assignment, cluster_map=None):
        self.assignment = assignment
        
        if cluster_map:
            my_start, my_end = assignment["start_layer"], assignment["end_layer"]
            for pid, shard in cluster_map.items():
                if shard["start_layer"] == my_end:
                    self.next_peer = pid
                if shard["end_layer"] == my_start:
                    self.prev_peer = pid
        
        if not self.engine:
            self.log("Initializing model shard...")
            self.engine = ShardEngine(
                model_name=assignment["model_id"],
                start_layer=assignment["start_layer"],
                end_layer=assignment["end_layer"],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            worker_state["status"] = "Ready"
            worker_state["layer_range"] = f"L{assignment['start_layer']}-{assignment['end_layer']}"

    def poll_and_process(self):
        try:
            res = requests.get(f"{COORDINATOR_URL}/relay/poll/{self.peer_id}")
            if res.status_code == 200 and res.content:
                msg_type = res.headers.get("X-Msg-Type", "activation")
                req_id = res.headers.get("X-Req-Id", "unknown")
                
                buffer = io.BytesIO(res.content)
                tensor = torch.load(buffer, map_location=self.engine.device, weights_only=False)
                
                if msg_type == "activation":
                    output = self.engine.forward_shard(tensor)
                    worker_state["processed"] += 1
                    
                    if self.next_peer:
                        out_buffer = io.BytesIO()
                        torch.save(output.detach().cpu(), out_buffer)
                        requests.post(
                            f"{COORDINATOR_URL}/relay/send/{self.next_peer}",
                            params={"msg_type": "activation", "req_id": req_id},
                            data=out_buffer.getvalue()
                        )
                    else:
                        # Sink - compute loss and backward
                        out_buffer = io.BytesIO()
                        torch.save(output.detach().cpu(), out_buffer)
                        requests.post(f"{COORDINATOR_URL}/relay/send/result_sink", data=out_buffer.getvalue())
                        
                        loss = output.pow(2).mean()
                        self.log(f"Loss: {loss.item():.4f}")
                        requests.post(f"{COORDINATOR_URL}/relay/send/loss_sink", data=str(loss.item()).encode())
                        
                        loss.backward()
                        input_grad = self.engine.last_input.grad
                        if input_grad is not None and self.prev_peer:
                            grad_buffer = io.BytesIO()
                            torch.save(input_grad.cpu(), grad_buffer)
                            requests.post(
                                f"{COORDINATOR_URL}/relay/send/{self.prev_peer}",
                                params={"msg_type": "gradient", "req_id": req_id},
                                data=grad_buffer.getvalue()
                            )
                        
                        self.engine.step()
                        requests.post(f"{COORDINATOR_URL}/relay/send/backward_sink", data=b"BACKWARD_DONE")
                
                elif msg_type == "gradient":
                    input_grad = self.engine.backward_shard(tensor)
                    if input_grad is not None and self.prev_peer:
                        grad_buffer = io.BytesIO()
                        torch.save(input_grad, grad_buffer)
                        requests.post(
                            f"{COORDINATOR_URL}/relay/send/{self.prev_peer}",
                            params={"msg_type": "gradient", "req_id": req_id},
                            data=grad_buffer.getvalue()
                        )
                    else:
                        requests.post(f"{COORDINATOR_URL}/relay/send/backward_sink", data=b"BACKWARD_DONE")
                    
                    self.engine.step()
        except:
            pass

# ============ Local Dashboard ============
dashboard_app = FastAPI(title="hdg@home Worker")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>hdg@home Worker</title>
<style>
body{font-family:system-ui;background:#0a0a0f;color:#f1f5f9;padding:2rem;max-width:600px;margin:0 auto}
.card{background:rgba(20,20,30,0.8);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:1.5rem;margin-bottom:1rem}
.title{color:#6366f1;font-size:1.5rem;font-weight:700;margin-bottom:1rem}
.stat{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#f1f5f9,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.label{color:#94a3b8;font-size:0.8rem;text-transform:uppercase}
.logs{background:#12121a;border-radius:8px;padding:1rem;font-family:monospace;font-size:0.75rem;max-height:300px;overflow-y:auto;color:#94a3b8}
</style></head><body>
<h1 class="title">🖥️ hdg@home Worker</h1>
<div class="card"><div class="stat" id="status">—</div><div class="label">Status</div></div>
<div class="card"><div class="stat" id="layers">—</div><div class="label">Layer Range</div></div>
<div class="card"><div class="stat" id="processed">0</div><div class="label">Batches Processed</div></div>
<div class="card"><div class="label">Live Logs</div><div class="logs" id="logs"></div></div>
<script>
async function update(){
  try{
    const r=await fetch('/state');const d=await r.json();
    document.getElementById('status').textContent=d.status;
    document.getElementById('layers').textContent=d.layer_range;
    document.getElementById('processed').textContent=d.processed;
    document.getElementById('logs').innerHTML=d.logs.slice(-20).map(l=>'<div>'+l+'</div>').join('');
  }catch(e){}
}
update();setInterval(update,1000);
</script></body></html>
"""

@dashboard_app.get("/")
def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

@dashboard_app.get("/state")
def state():
    worker_state["ram_percent"] = psutil.virtual_memory().percent
    return JSONResponse(worker_state)

def run_dashboard():
    uvicorn.run(dashboard_app, host="0.0.0.0", port=8080, log_level="warning")

# ============ Main ============
if __name__ == "__main__":
    # Start local dashboard
    threading.Thread(target=run_dashboard, daemon=True).start()
    print(f"Local dashboard: http://localhost:8080\n")
    
    # Run worker
    client = WorkerClient()
    client.loop()
