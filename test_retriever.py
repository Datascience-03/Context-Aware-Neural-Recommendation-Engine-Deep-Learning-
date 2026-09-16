import numpy as np
import redis
from retriever import VectorRetriever

# Try connecting to real Redis; fallback to FakeRedis if Redis server isn't running
try:
    r = redis.Redis(host="localhost", port=6379, db=0)
    r.ping()
    print("[INFO] Connected to real Redis server.")
except (redis.exceptions.ConnectionError, ConnectionRefusedError):
    import fakeredis

    r = fakeredis.FakeRedis()
    print("[INFO] Real Redis not running. Using FakeRedis (in-memory mock).")

if __name__ == "__main__":
    retriever = VectorRetriever(redis_client=r, key_prefix="item:")

    # 1. Populate dummy float32 vectors
    dim = 4
    vec_101 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    vec_102 = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)

    r.set("item:101", vec_101.tobytes())
    r.set("item:102", vec_102.tobytes())

    # 2. Test Single Item Fetch
    print("\n--- 1. Testing Single Fetch ---")
    retrieved_101 = retriever.get_vector(101)
    print("Item 101:", retrieved_101)
    print("Data type:", retrieved_101.dtype)

    # 3. Test Missing Value Handling
    print("\n--- 2. Testing Missing Key Handling ---")
    missing_none = retriever.get_vector(999)  # Returns None
    print("Missing Item (Default fallback):", missing_none)

    missing_zero = retriever.get_vector(999, dim=dim, fallback_to_zeros=True)
    print("Missing Item (Zero fallback):", missing_zero)

    # 4. Test Multi-Item Fetch (MGET)
    print("\n--- 3. Testing Batch Fetch (MGET) ---")
    query_ids = [101, 999, 102]
    batch_results = retriever.get_vectors_batch(
        item_ids=query_ids, dim=dim, fallback_to_zeros=True
    )

    for item_id, vec in zip(query_ids, batch_results):
        print(f"ID {item_id} -> {vec}")