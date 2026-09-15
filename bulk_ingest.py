import time
import logging
import json
import redis
from typing import Generator, Dict, Any

# 1. Setup Logging for Monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("BulkIngestion")

# 2. Redis Connection Settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
BATCH_SIZE = 2000  # Chunks of 1,000–5,000 items

def get_redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )

def generate_sample_catalog(total_items: int = 100_000) -> Generator[Dict[str, Any], None, None]:
    """
    Simulates streaming/reading the full item catalog.
    Replace this with reading from your CSV/JSON/Database file.
    """
    for i in range(1, total_items + 1):
        yield {
            "item_id": f"item:{i}",
            "data": {
                "name": f"Product {i}",
                "price": round(10.0 + (i % 100) * 1.5, 2),
                "category": f"Category_{i % 10}",
                "stock": 50 + (i % 200)
            }
        }

def bulk_ingest(client: redis.Redis, dataset_generator, batch_size: int = BATCH_SIZE):
    """
    Ingests items into Redis using pipelining in configurable batches.
    """
    logger.info(f"Starting bulk ingestion with batch size: {batch_size}...")
    
    total_items = 0
    start_time = time.perf_counter()
    batch_start_time = start_time
    
    # Initialize pipeline with transaction=False for maximum throughput
    pipe = client.pipeline(transaction=False)
    
    for item in dataset_generator:
        key = item["item_id"]
        val = item["data"]
        
        # Option A: Store as Redis Hash (recommended for structured catalogs)
        pipe.hset(key, mapping=val)
        
        # Option B: Alternatively, store as JSON/String:
        # pipe.set(key, json.dumps(val))
        
        total_items += 1

        # When batch size is reached, execute the pipeline
        if total_items % batch_size == 0:
            pipe.execute()
            
            # Monitoring checkpoint
            batch_duration = time.perf_counter() - batch_start_time
            batch_rate = batch_size / batch_duration
            logger.info(
                f"Ingested {total_items:,} items | Current Batch Rate: {batch_rate:,.2f} items/sec"
            )
            batch_start_time = time.perf_counter()

    # Flush any remaining commands left in the pipeline buffer
    if len(pipe) > 0:
        remaining_count = len(pipe)
        pipe.execute()
        logger.info(f"Flushed final {remaining_count} remaining items.")

    # Final Summary Logging
    total_duration = time.perf_counter() - start_time
    overall_rate = total_items / total_duration if total_duration > 0 else 0

    logger.info("=" * 50)
    logger.info("BULK INGESTION COMPLETE")
    logger.info(f"Total Items Written: {total_items:,}")
    logger.info(f"Total Duration     : {total_duration:.2f} seconds")
    logger.info(f"Average Throughput : {overall_rate:,.2f} items/sec")
    logger.info("=" * 50)

if __name__ == "__main__":
    try:
        r = get_redis_client()
        r.ping()
        logger.info("Connected to Redis successfully.")
        
        # Ingest 100,000 items (adjust to your full catalog size or load a file)
        catalog = generate_sample_catalog(total_items=100_000)
        bulk_ingest(r, catalog, batch_size=BATCH_SIZE)

    except redis.ConnectionError as e:
        logger.error(f"Could not connect to Redis: {e}")