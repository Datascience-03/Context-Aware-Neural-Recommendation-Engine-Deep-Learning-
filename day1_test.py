import redis
import numpy as np

# 1. Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

# Verify connection
try:
    r.ping()
    print(" Successfully connected to Redis!")
except redis.ConnectionError:
    print(" Connection failed. Make sure Redis is running.")
    exit(1)

# 2. Define schema constants
DIMENSION = 128  # Example dimension
ITEM_ID = 42
KEY = f"item:vec:{ITEM_ID}"

# 3. Create a random mock vector (FP32)
original_vector = np.random.rand(DIMENSION).astype(np.float32)

# 4. Serialize to raw bytes
serialized_vector = original_vector.tobytes()
print(f"\nSerialized payload size: {len(serialized_vector)} bytes (expected: {DIMENSION * 4} bytes)")

# 5. Store in Redis
r.set(KEY, serialized_vector)
print(f"Stored vector under key: '{KEY}'")

# 6. Retrieve from Redis
raw_data_from_redis = r.get(KEY)

# 7. Deserialize back into NumPy array
retrieved_vector = np.frombuffer(raw_data_from_redis, dtype=np.float32)

# 8. Assert integrity
assert np.array_equal(original_vector, retrieved_vector), "Data mismatch!"
print(" Vector integrity check passed: Retrieved data matches original!")

# Clean up
r.delete(KEY)