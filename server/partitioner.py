from typing import List, Dict, Any
from pydantic import BaseModel
import math

class ModelShardDef(BaseModel):
    peer_id: str
    start_layer: int
    end_layer: int
    total_layers: int
    model_id: str

class Partitioner:
    def __init__(self, model_config: Dict[str, Any]):
        """
        model_config: Dict containing model details, e.g. 
        {"num_hidden_layers": 32, "hidden_size": 4096, ...}
        """
        self.config = model_config
        self.num_layers = model_config.get("num_hidden_layers", 12) # Default to small if unknown

    def partition(self, active_peers: List[Dict[str, Any]]) -> Dict[str, ModelShardDef]:
        """
        Naive partitioner: splits layers evenly among available peers.
        In future: use peer['capabilities'] to weight the split.
        """
        if not active_peers:
            return {}

        num_peers = len(active_peers)
        layers_per_peer = math.ceil(self.num_layers / num_peers)
        
        assignment = {}
        current_layer = 0
        
        for i, peer in enumerate(active_peers):
            peer_id = peer['peer_id']
            # Determine range
            end = min(current_layer + layers_per_peer, self.num_layers)
            
            # If we run out of layers, some peers might get idle (or redundant - pending design)
            # For now, if we are out of layers, assign empty or duplicate? 
            # Let's just stop assigning if we run out.
            if current_layer >= self.num_layers:
                break

            assignment[peer_id] = ModelShardDef(
                peer_id=peer_id,
                start_layer=current_layer,
                end_layer=end,
                total_layers=self.num_layers,
                model_id=self.config.get("model_id", "unknown")
            )
            current_layer = end
            
        return assignment
