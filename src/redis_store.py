import redis
import numpy as np
from typing import Optional

class VectorStore:
    """
    Day 1 Schema Implementation:
    - Key convention: item:vec:<item_id>
    - Serialization: Raw FP32 binary bytes
    """
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def _format_key(self, item_id: str | int) -> str:
        return f"item:vec:{item_id}"

    def set_vector(self, item_id: str | int, vector: np.ndarray) -> bool:
        """Serializes and saves a vector using raw bytes (FP32)."""
        key = self._format_key(item_id)
        # Ensure array is float32 for consistent sizing and fast serialization
        raw_bytes = vector.astype(np.float32).tobytes()
        return self.client.set(key, raw_bytes)

    def get_vector(self, item_id: str | int) -> Optional[np.ndarray]:
        """Fetches and deserializes a vector from raw bytes."""
        key = self._format_key(item_id)
        raw_bytes = self.client.get(key)
        if raw_bytes is None:
            return None
        return np.frombuffer(raw_bytes, dtype=np.float32)

    def ping(self) -> bool:
        return self.client.ping()