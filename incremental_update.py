import asyncio
import random
from typing import Dict, Any
from client_wrapper import FaultTolerantVectorClient

async def process_catalog_event(
    client: FaultTolerantVectorClient, event: Dict[str, Any]
):
    """
    Simulates consuming a catalog event (e.g., from Kafka/RabbitMQ/Webhook)
    and updating the vector index incrementally.
    """
    item_id = event.get("item_id")
    vector = event.get("vector")

    if not item_id or not vector:
        print(f"[SKIP] Invalid event payload: {event}")
        return

    try:
        success = await client.upsert_vector(item_id, vector)
        if success:
            print(f"[UPSERT SUCCESS] Item: {item_id} (Dims: {len(vector)})")
    except Exception as e:
        print(f"[UPSERT ERROR] Failed to update item {item_id}: {e}")

async def main():
    client = FaultTolerantVectorClient()

    # Incoming catalog stream simulation
    catalog_stream = [
        {"item_id": f"item_new_{i}", "vector": [random.random() for _ in range(128)]}
        for i in range(5)
    ]

    print("Processing incoming catalog updates...")
    await asyncio.gather(*(process_catalog_event(client, item) for item in catalog_stream))

    # Verify updated items via lookup
    test_ids = [item["item_id"] for item in catalog_stream]
    results = await client.batch_lookup(test_ids)
    print(f"\nVerification lookup: successfully retrieved {len(results)} items.")

if __name__ == "__main__":
    asyncio.run(main())