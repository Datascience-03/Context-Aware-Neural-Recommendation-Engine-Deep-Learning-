from typing import List, Optional, Union
import numpy as np
import redis


class VectorRetrievalModule:

    def __init__(
        self,
        redis_client: redis.Redis,
        vector_dim: int = 128,
        key_prefix: str = "item:",
    ):
        """Initialize the vector retrieval module.

        Args:
            redis_client: An active redis.Redis connection client.
            vector_dim: Expected dimension of vectors (used when returning
            zero-vectors).
            key_prefix: Optional prefix used when storing items in Redis (e.g.,
            'item:101').
        """
        self.r = redis_client
        self.vector_dim = vector_dim
        self.key_prefix = key_prefix

    def _format_key(self, item_id: Union[str, int]) -> str:
        """Format the Redis key with prefix."""
        return f"{self.key_prefix}{item_id}"

    # -------------------------------------------------------------
    # 1. Single Item Fetch
    # -------------------------------------------------------------
    def get_vector(
        self, item_id: Union[str, int], fallback_to_zero: bool = False
    ) -> Optional[np.ndarray]:
        """Fetch a single vector by item ID.

        Args:
            item_id: Unique identifier for the item.
            fallback_to_zero: If True, returns np.zeros when key is missing;
              otherwise returns None.

        Returns:
            np.ndarray (dtype=np.float32) or None/zero-vector if not found.
        """
        key = self._format_key(item_id)
        raw_bytes = self.r.get(key)

        if raw_bytes is None:
            return (
                np.zeros(self.vector_dim, dtype=np.float32)
                if fallback_to_zero
                else None
            )

        # Reconstruct vector directly from byte buffer
        return np.frombuffer(raw_bytes, dtype=np.float32).copy()

    # -------------------------------------------------------------
    # 2. Multi-Item Fetch (MGET) & 3. Missing Value Handling
    # -------------------------------------------------------------
    def get_vectors_batch(
        self,
        item_ids: List[Union[str, int]],
        fallback_to_zero: bool = False,
    ) -> List[Optional[np.ndarray]]:
        """Fetch multiple item vectors using Redis MGET in a single round-trip.

        Args:
            item_ids: List of item identifiers.
            fallback_to_zero: If True, replaces missing vectors with a zero-vector.
              If False, puts None in place of missing vectors.

        Returns:
            List of np.ndarray vectors (or None/zero-vector for missing items),
            preserving the input order.
        """
        if not item_ids:
            return []

        # Generate keys for batch lookup
        keys = [self._format_key(item_id) for item_id in item_ids]

        # Single network round-trip using MGET
        raw_bytes_list = self.r.mget(keys)

        results: List[Optional[np.ndarray]] = []
        zero_fallback = (
            np.zeros(self.vector_dim, dtype=np.float32)
            if fallback_to_zero
            else None
        )

        for raw_bytes in raw_bytes_list:
            if raw_bytes is None:
                # Safe fallback: don't crash on missing IDs
                results.append(
                    zero_fallback.copy() if fallback_to_zero else None
                )
            else:
                vector = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                results.append(vector)

        return results

    def get_vectors_batch_as_matrix(
        self, item_ids: List[Union[str, int]]
    ) -> np.ndarray:
        """Convenience method for ranking pipelines:

        Returns a 2D matrix of shape (N, vector_dim) with missing items
        imputed as zeros.
        """
        vectors = self.get_vectors_batch(item_ids, fallback_to_zero=True)
        return np.vstack(vectors)