from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from enum import Enum
import time

class PeerType(str, Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"

class Capability(BaseModel):
    vram_mb: int
    compute_score: float  # Normalized FLOPS or similar metric
    bandwidth_mbps: float
    peer_id: str

class TaskType(str, Enum):
    TRAIN_STEP = "train_step"
    VALIDATION_STEP = "validation_step"
    PING = "ping"

class TaskAssignment(BaseModel):
    task_id: str
    task_type: TaskType
    model_shard_id: str  # Identifier for the specific layers/shard
    base_model_version: str # Hash or version tag
    data_batch_id: str
    config: Dict[str, Any] = Field(default_factory=dict)

class TrainingResult(BaseModel):
    task_id: str
    peer_id: str
    success: bool
    gradients_hash: Optional[str] = None # Hash of the gradients for verification
    metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    error_message: Optional[str] = None
    
class PeerHeartbeat(BaseModel):
    peer_id: str
    capabilities: Capability
    status: str # "idle", "busy", "offline"
    timestamp: float = Field(default_factory=time.time)
