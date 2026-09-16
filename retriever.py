# retriever.py
from typing import List, Optional, Union
import numpy as np
import redis


class VectorRetriever:

    def __init__(self, redis_client: redis.Redis, key_prefix: str = "item:"):
        self.client = redis_client
        self.key_prefix = key_prefix

    def _get_key(self, item_id: Union[str, int]) -> str:
        return f"{self.key_prefix}{item_id}"

    def get_vector(
        self,
        item_id: Union[str, int],
        dim: Optional[int] = None,
        fallback_to_zeros: bool = False,
    ) -> Optional[np.ndarray]:
        key = self._get_key(item_id)
        raw_bytes = self.client.get(key)
        if raw_bytes is None:
            if fallback_to_zeros and dim is not None:
                return np.zeros(dim, dtype=np.float32)
            return None
        return np.frombuffer(raw_bytes, dtype=np.float32)

    def get_vectors_batch(
        self,
        item_ids: List[Union[str, int]],
        dim: Optional[int] = None,
        fallback_to_zeros: bool = False,
    ) -> List[Optional[np.ndarray]]:
        if not item_ids:
            return []
        keys = [self._get_key(item_id) for item_id in item_ids]
        raw_results = self.client.mget(keys)

        vectors: List[Optional[np.ndarray]] = []
        for raw_bytes in raw_results:
            if raw_bytes is None:
                if fallback_to_zeros and dim is not None:
                    vectors.append(np.zeros(dim, dtype=np.float32))
                else:
                    vectors.append(None)
            else:
                vectors.append(np.frombuffer(raw_bytes, dtype=np.float32))
        return vectors