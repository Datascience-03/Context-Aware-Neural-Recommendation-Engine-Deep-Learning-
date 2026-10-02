import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VectorStoreClient")


class FaultTolerantVectorClient:
    def __init__(
        self,
        timeout: float = 0.05,  # 50ms total timeout
        max_retries: int = 3,
        base_backoff: float = 0.01,  # 10ms initial backoff
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        # In-memory mock storage (replace with your real DB client instance, e.g. redis.Redis)
        self._db: Dict[str, List[float]] = {
            f"item_{i}": [random.random() for _ in range(128)] for i in range(1000)
        }

    async def ping(self) -> bool:
        """Health-check method to ensure the DB connection is alive."""
        try:
            # Replace with: await self.redis_client.ping() or db.is_healthy()
            await asyncio.sleep(0.0005)  # Simulated 0.5ms network round-trip
            return True
        except Exception as e:
            logger.error(f"Health-check failed: {e}")
            return False

    async def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Executes an operation with timeout, retries, and exponential jitter backoff."""
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Enforce strict request timeout
                return await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.timeout
                )
            except (asyncio.TimeoutError, ConnectionError, Exception) as exc:
                last_exception = exc
                if attempt == self.max_retries:
                    break

                # Full-jitter exponential backoff
                jitter = random.uniform(0, self.base_backoff * (2 ** (attempt - 1)))
                logger.warning(
                    f"Attempt {attempt} failed ({type(exc).__name__}). Retrying in {jitter*1000:.2f}ms..."
                )
                await asyncio.sleep(jitter)

        logger.error(f"Operation failed after {self.max_retries} attempts.")
        raise last_exception

    async def _raw_batch_lookup(self, ids: List[str]) -> Dict[str, Optional[List[float]]]:
        """Simulates/Performs low-level batch lookup."""
        await asyncio.sleep(0.001)  # Simulated DB latency ~1ms
        return {item_id: self._db.get(item_id) for item_id in ids}

    async def batch_lookup(self, ids: List[str]) -> Dict[str, Optional[List[float]]]:
        """Fault-tolerant batch lookup."""
        return await self._execute_with_retry(self._raw_batch_lookup, ids)

    async def _raw_upsert(self, item_id: str, vector: List[float]) -> bool:
        """Simulates/Performs vector insertion or update."""
        await asyncio.sleep(0.0015)  # Simulated DB write latency
        self._db[item_id] = vector
        return True

    async def upsert_vector(self, item_id: str, vector: List[float]) -> bool:
        """Fault-tolerant upsert."""
        return await self._execute_with_retry(self._raw_upsert, item_id, vector)