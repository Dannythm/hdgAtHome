from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uuid
import time
import secrets
import hashlib
# from ..common.protocol import PeerType, Capability, TaskAssignment, PeerHeartbeat

# Hack to fix import for now until package structure is set
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.protocol import PeerType, Capability, TaskAssignment, PeerHeartbeat, TaskType

app = FastAPI(title="hdg@home Coordinator")
security = HTTPBasic()

# ============ CONFIGURATION ============
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "hdghome2024"  # Change this in production!
# =======================================

# Mount Static Dashboard (will add auth middleware)
static_path = os.path.join(os.path.dirname(__file__), 'static')

# State
peers: Dict[str, PeerHeartbeat] = {}
assignments: Dict[str, Any] = {} # peer_id -> ModelShardDef
tensor_mailbox: Dict[str, List[Dict]] = {} # peer_id -> List[{data_id, blob, metadata}]

# Worker API Keys: api_key -> {peer_id, created_at, nickname}
worker_api_keys: Dict[str, Dict] = {}

# Config
MODEL_NAME = "H:/gen_ai/llm/gpt2"
# MODEL_NAME = "gpt2"
from .partitioner import Partitioner
partitioner = None 

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials for dashboard access"""
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def verify_worker_key(api_key: str) -> Optional[Dict]:
    """Verify worker API key"""
    if api_key in worker_api_keys:
        return worker_api_keys[api_key]
    return None

@app.on_event("startup")
def startup_event():
    global partitioner
    # Initialize partitioner with default model config
    # In real world, we'd load this from args or a config file
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME)
    partitioner = Partitioner({
        "num_hidden_layers": config.num_hidden_layers,
        "model_id": MODEL_NAME
    })
    print(f"Coordinator started for model: {MODEL_NAME}")

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "active_peers": len(peers),
        "assignments": {k: v.dict() for k,v in assignments.items()}
    }

@app.get("/join")
def join_page():
    """Public landing page for new contributors"""
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_path, "join.html"))

@app.get("/dashboard/")
def dashboard_page(admin: str = Depends(verify_admin)):
    """Protected admin dashboard"""
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(static_path, "index.html"))

@app.get("/dashboard/{path:path}")
def dashboard_static(path: str, admin: str = Depends(verify_admin)):
    """Protected static files for dashboard"""
    from fastapi.responses import FileResponse
    file_path = os.path.join(static_path, path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/worker.py")
def download_worker():
    """Download standalone worker script"""
    from fastapi.responses import FileResponse
    return FileResponse(
        os.path.join(static_path, "worker.py"),
        filename="worker.py",
        media_type="text/x-python"
    )

# ============ API Key Management (Admin) ============

class APIKeyRequest(BaseModel):
    nickname: Optional[str] = None

@app.post("/admin/generate-key")
def generate_api_key(request: APIKeyRequest = None, admin: str = Depends(verify_admin)):
    """Generate a new API key for a worker (Admin only)"""
    api_key = secrets.token_urlsafe(32)
    worker_api_keys[api_key] = {
        "created_at": time.time(),
        "nickname": request.nickname if request else None,
        "peer_id": None  # Will be set on first registration
    }
    return {"api_key": api_key, "nickname": request.nickname if request else None}

@app.get("/admin/list-keys")
def list_api_keys(admin: str = Depends(verify_admin)):
    """List all API keys (Admin only)"""
    return {
        "keys": [
            {
                "key_preview": k[:8] + "...",
                "nickname": v.get("nickname"),
                "peer_id": v.get("peer_id"),
                "created_at": v.get("created_at")
            }
            for k, v in worker_api_keys.items()
        ]
    }

@app.delete("/admin/revoke-key/{key_preview}")
def revoke_api_key(key_preview: str, admin: str = Depends(verify_admin)):
    """Revoke an API key (Admin only)"""
    for k in list(worker_api_keys.keys()):
        if k.startswith(key_preview):
            del worker_api_keys[k]
            return {"status": "revoked", "key_preview": key_preview}
    raise HTTPException(status_code=404, detail="Key not found")

@app.post("/register")
def register_peer(capability: Capability):
    peer_id = str(uuid.uuid4()) if not capability.peer_id else capability.peer_id
    peers[peer_id] = PeerHeartbeat(
        peer_id=peer_id,
        capabilities=capability,
        status="idle",
        timestamp=time.time()
    )
    # Init mailbox
    if peer_id not in tensor_mailbox:
        tensor_mailbox[peer_id] = []
        
    # Trigger Re-partition logic if we have enough peers?
    # For prototype: partition whenever we have 2 peers and no assignments
    if len(peers) >= 2 and not assignments:
        print("Triggering Partitioning...")
        peer_list = [{"peer_id": p, "capabilities": {}} for p in peers]
        new_assignments = partitioner.partition(peer_list)
        for pid, shard_def in new_assignments.items():
            assignments[pid] = shard_def
            print(f"Assigned {pid} -> Layers {shard_def.start_layer}-{shard_def.end_layer}")
            
    return {"peer_id": peer_id, "message": "Registered successfully"}

@app.post("/heartbeat")
def heartbeat(hb: PeerHeartbeat):
    if hb.peer_id not in peers:
        raise HTTPException(status_code=404, detail="Peer not registered")
    
    peers[hb.peer_id] = hb
    peers[hb.peer_id].timestamp = time.time()
    
    # Debug
    print(f"HB from {hb.peer_id}. Peers: {len(peers)}, Assignments: {len(assignments)}")
    
    # Check assignment
    if hb.peer_id in assignments:
        # If peer is idle, tell it to initialize or process
        # We need a state tracking if peer has accepted assignment
        return {
            "action": "assign_shard",
            "assignment": assignments[hb.peer_id].dict(),
            "cluster_map": {k: v.dict() for k, v in assignments.items()}
        }
    
    return {"action": "wait"}

@app.post("/relay/push")
async def push_tensor(
    target_peer_id: str, 
    metadata: str, # JSON string
    file: bytes = None # For fastAPI file upload, actually better to use UploadFile
):
    # Simplified for Pydantic/JSON proto: receive base64 or bytes?
    # FastAPI File upload is best for large blobs
    # For now, let's assume client sends bytes as body or form
    pass 
    
# Better implementation for Tensor Relay using simple dict for now
# Client will POST arbitrary binary data?
from fastapi import Request
@app.post("/relay/send/{target_peer_id}")
async def send_tensor(
    target_peer_id: str, 
    request: Request,
    msg_type: str = "activation", # activation | gradient
    req_id: str = "default" # Batch ID
):
    """
    Receive binary tensor data destined for target_peer_id.
    """
    data = await request.body()
    if target_peer_id not in tensor_mailbox:
        tensor_mailbox[target_peer_id] = []
    
    # Store simplified
    tensor_mailbox[target_peer_id].append({
        "timestamp": time.time(),
        "data": data,
        "size": len(data),
        "msg_type": msg_type,
        "req_id": req_id
    })
    return {"status": "buffered", "queue_size": len(tensor_mailbox[target_peer_id])}

@app.get("/relay/poll/{peer_id}")
def poll_tensor(peer_id: str):
    """
    Peer asks: do I have data?
    """
    if peer_id in tensor_mailbox and tensor_mailbox[peer_id]:
        # Pop one item (FIFO)
        item = tensor_mailbox[peer_id].pop(0)
        from fastapi import Response
        res = Response(content=item["data"], media_type="application/octet-stream")
        # Add metadata as headers
        res.headers["X-Msg-Type"] = item.get("msg_type", "activation")
        res.headers["X-Req-Id"] = item.get("req_id", "default")
        return res
    
    from fastapi import Response
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
