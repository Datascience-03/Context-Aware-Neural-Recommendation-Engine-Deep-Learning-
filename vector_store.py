import os
from typing import Dict, List, Optional
import numpy as np
import redis

try:
    import fakeredis
except ImportError:
    fakeredis = None


class ItemVectorStore:
    """Production-ready client SDK to read and write item embedding vectors in Redis."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        key_prefix: str = "item:vector:",
        dtype: np.dtype = np.float32,
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", 6379))
        self.db = db or int(os.getenv("REDIS_DB", 0))
        self.password = password or os.getenv("REDIS_PASSWORD", None)

        self.key_prefix = key_prefix
        self.dtype = dtype

        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False,
                socket_connect_timeout=1,
            )
            self.client.ping()
            print("Connected to Live Redis server.")
        except Exception:
            if fakeredis is not None:
                print("Live Redis not reachable. Using in-memory FakeRedis.")
                self.client = fakeredis.FakeRedis()
            else:
                raise ConnectionError("Redis is offline and fakeredis is not installed.")

    def _format_key(self, item_id: str) -> str:
        return f"{self.key_prefix}{item_id}"

    def set_vector(self, item_id: str, vector: np.ndarray) -> None:
        key = self._format_key(item_id)
        raw_bytes = vector.astype(self.dtype).tobytes()
        self.client.set(key, raw_bytes)

    def set_vectors_batch(self, items: Dict[str, np.ndarray]) -> None:
        if not items:
            return
        pipeline = self.client.pipeline()
        for item_id, vector in items.items():
            key = self._format_key(item_id)
            raw_bytes = vector.astype(self.dtype).tobytes()
            pipeline.set(key, raw_bytes)
        pipeline.execute()

    def get_vector(self, item_id: str) -> Optional[np.ndarray]:
        key = self._format_key(item_id)
        raw_bytes = self.client.get(key)
        if raw_bytes is None:
            return None
        return np.frombuffer(raw_bytes, dtype=self.dtype).copy()

    def get_vectors_batch(self, item_ids: List[str]) -> Dict[str, np.ndarray]:
        if not item_ids:
            return {}
        keys = [self._format_key(i) for i in item_ids]
        raw_records = self.client.mget(keys)

        results: Dict[str, np.ndarray] = {}
        for item_id, raw_bytes in zip(item_ids, raw_records):
            if raw_bytes is not None:
                results[item_id] = np.frombuffer(raw_bytes, dtype=self.dtype).copy()
        return results

    def delete_vector(self, item_id: str) -> bool:
        return bool(self.client.delete(self._format_key(item_id)))

    def ping(self) -> bool:
        return bool(self.client.ping())