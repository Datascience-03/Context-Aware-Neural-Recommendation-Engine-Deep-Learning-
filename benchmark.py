import asyncio
import numpy as np
import random
import time
from client_wrapper import FaultTolerantVectorClient

async def worker(
    client: FaultTolerantVectorClient,
    batch_size: int,
    num_requests: int,
    latencies: list,
    semaphore: asyncio.Semaphore,
):
    """Worker task sending batch lookup queries under concurrency limit."""
    for _ in range(num_requests):
        sample_ids = [f"item_{random.randint(0, 999)}" for _ in range(batch_size)]
        
        async with semaphore:
            start_time = time.perf_counter()
            try:
                await client.batch_lookup(sample_ids)
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                latencies.append(latency_ms)
            except Exception as e:
                # Log errors in real benchmarks
                pass

async def run_benchmark(
    concurrency_limit: int = 100,
    total_workers: int = 50,
    requests_per_worker: int = 40,
    batch_size: int = 10,
):
    client = FaultTolerantVectorClient()

    # 1. Health-check verification
    print("Checking database health status...")
    if not await client.ping():
        print("[ERROR] Database health check failed. Aborting benchmark.")
        return
    print("[OK] Database is healthy.\n")

    print(f"Starting Benchmark:")
    print(f"  - Concurrency Limit: {concurrency_limit}")
    print(f"  - Total Requests:    {total_workers * requests_per_worker}")
    print(f"  - Batch Size:        {batch_size} vectors per request\n")

    latencies = []
    semaphore = asyncio.Semaphore(concurrency_limit)

    start_total = time.perf_counter()
    tasks = [
        worker(client, batch_size, requests_per_worker, latencies, semaphore)
        for _ in range(total_workers)
    ]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_total

    # Calculate percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    qps = len(latencies) / total_time

    print("================ Benchmark Results ================")
    print(f"Throughput: {qps:.2f} QPS")
    print(f"P50 Latency: {p50:.2f} ms")
    print(f"P95 Latency: {p95:.2f} ms")
    print(f"P99 Latency: {p99:.2f} ms")
    print("====================================================")

    if p99 < 5.0:
        print("[PASS] Target achieved: Sub-5ms batch read latency.")
    else:
        print("[FAIL] Target missed: P99 latency exceeded 5ms.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())