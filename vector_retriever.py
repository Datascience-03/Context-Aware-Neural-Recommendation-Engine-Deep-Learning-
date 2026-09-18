import numpy as np
import redis
from typing import List, Optional, Union, Dict

class VectorRetriever:
    """
    Fast vector retrieval module from Redis using raw byte storage and NumPy.
    """
    def __init__(self, redis_client: redis.Redis, key_prefix: str = "item:"):
        """
        :param redis_client: An active redis.Redis connection instance.
        :param key_prefix: Prefix used for item keys (e.g., 'item:123').
        """
        self.client = redis_client
        self.key_prefix = key_prefix

    def _get_key(self, item_id: Union[str, int]) -> str:
        """Helper method to format the Redis key."""
        return f"{self.key_prefix}{item_id}"

    # -------------------------------------------------------------
    # 1. Single Item Fetch
    # -------------------------------------------------------------
    def get_vector(
        self,
        item_id: Union[str, int],
        dim: Optional[int] = None,
        fallback_to_zeros: bool = False
    ) -> Optional[np.ndarray]:
        """
        Fetch a single item vector by its ID.

        :param item_id: Unique identifier for the item.
        :param dim: Dimensionality of the vector (required if fallback_to_zeros=True).
        :param fallback_to_zeros: If True, returns np.zeros(dim) for missing keys.
        :return: NumPy array (float32) or None/zero-vector if not found.
        """
        key = self._get_key(item_id)
        raw_bytes = self.client.get(key)

        # Missing Value Handling
        if raw_bytes is None:
            if fallback_to_zeros and dim is not None:
                return np.zeros(dim, dtype=np.float32)
            return None

        # Reconstruct vector directly from bytes (O(1) memory view)
        return np.frombuffer(raw_bytes, dtype=np.float32)

    # -------------------------------------------------------------
    # 2. Multi-Item Fetch (MGET)
    # -------------------------------------------------------------
    def get_vectors_batch(
        self,
        item_ids: List[Union[str, int]],
        dim: Optional[int] = None,
        fallback_to_zeros: bool = False
    ) -> List[Optional[np.ndarray]]:
        """
        Fetch multiple item vectors in a single round-trip using Redis MGET.

        :param item_ids: List of item IDs to fetch.
        :param dim: Vector dimensionality (required if fallback_to_zeros=True).
        :param fallback_to_zeros: If True, replaces missing items with zero-vectors.
        :return: Ordered list of NumPy arrays (or None / zero vectors).
        """
        if not item_ids:
            return []

        keys = [self._get_key(item_id) for item_id in item_ids]
        raw_results = self.client.mget(keys)

        vectors: List[Optional[np.ndarray]] = []
        for raw_bytes in raw_results:
            # Missing Value Handling
            if raw_bytes is None:
                if fallback_to_zeros and dim is not None:
                    vectors.append(np.zeros(dim, dtype=np.float32))
                else:
                    vectors.append(None)
            else:
                vectors.append(np.frombuffer(raw_bytes, dtype=np.float32))

        return vectors

    # -------------------------------------------------------------
    # Optional Helper: Batch fetch as Dictionary
    # -------------------------------------------------------------
    def get_vectors_dict(
        self,
        item_ids: List[Union[str, int]],
        dim: Optional[int] = None,
        fallback_to_zeros: bool = False
    ) -> Dict[Union[str, int], Optional[np.ndarray]]:
        """Convenience method returning a mapping of {item_id: vector}."""
        vectors = self.get_vectors_batch(item_ids, dim=dim, fallback_to_zeros=fallback_to_zeros)
        return dict(zip(item_ids, vectors))