import redis
import fakeredis
import numpy as np
import pandas as pd
import time
import os

# 1. Connection Pooling Configuration
pool = redis.ConnectionPool(
    connection_class=fakeredis.FakeConnection,
    max_connections=20,
    decode_responses=False
)

def get_redis_client():
    return redis.Redis(connection_pool=pool)

# 2. Vector Serialization Helper
def serialize_vector(vector: np.ndarray) -> bytes:
    return vector.astype(np.float32).tobytes()

def deserialize_vector(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, dtype=np.float32)

# 3. Ingestion Pipeline
def ingest_from_parquet(filepath: str, batch_size: int = 500):
    client = get_redis_client()
    print(f"Reading data from {filepath}...")
    df = pd.read_parquet(filepath)
    total_records = len(df)

    start_time = time.time()
    pipe = client.pipeline(transaction=False)

    count = 0
    for idx, row in df.iterrows():
        item_id = row["item_id"]
        vector_np = np.array(row["vector"], dtype=np.float32)
        vector_bytes = serialize_vector(vector_np)

        redis_key = f"item:{item_id}"
        pipe.hset(redis_key, mapping={"vector": vector_bytes})
        count += 1

        if count % batch_size == 0 or count == total_records:
            pipe.execute()
            print(f"Ingested {count}/{total_records} items...")

    elapsed = time.time() - start_time
    print(f"\nIngestion finished! {count} items ingested in {elapsed:.3f}s ({count/elapsed:.1f} items/sec).")

# 4. Verification Check
def verify_sample(sample_id="item_0"):
    client = get_redis_client()
    raw_vector = client.hget(f"item:{sample_id}", "vector")

    if raw_vector:
        vector = deserialize_vector(raw_vector)
        print(f"\n[Verification] Successfully retrieved {sample_id}:")
        print(f" - Vector shape: {vector.shape}")
        print(f" - Vector dtype: {vector.dtype}")
        print(f" - Vector sample (first 5 dims): {vector[:5]}")
    else:
        print(f"[Verification Error] Could not find item {sample_id}")

if __name__ == "__main__":
    DATA_FILE = "items_vectors.parquet"
    if not os.path.exists(DATA_FILE) and os.path.exists(os.path.join("src", "pipeline", DATA_FILE)):
        DATA_FILE = os.path.join("src", "pipeline", DATA_FILE)

    ingest_from_parquet(DATA_FILE)
    verify_sample("item_0")
