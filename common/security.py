import hashlib
import pickle

def compute_model_hash(model_state_dict) -> str:
    """
    Compute a SHA256 hash of the model state dict keys and values (roughly).
    In production this needs to be more robust and efficient.
    """
    hasher = hashlib.sha256()
    # This is a naive implementation, real tensors need careful serialization for consistent hashing
    for key in sorted(model_state_dict.keys()):
        hasher.update(key.encode('utf-8'))
        # For prototype we might skip hashing huge tensors every time or use a faster checksum
    return hasher.hexdigest()

def sign_payload(payload: bytes, private_key) -> bytes:
    # TODO: Implement ed25519 or similar
    pass
    
def verify_signature(payload: bytes, signature: bytes, public_key) -> bool:
    # TODO: Implement
    return True
