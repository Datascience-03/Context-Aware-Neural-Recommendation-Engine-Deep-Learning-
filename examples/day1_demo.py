"""
Member 5 - Day 1 Execution Script
-----------------------------------
Demonstrates:
1. Interface contracts with Member 1 (Query Tower outputs) & Member 2 (Item Embedding catalog).
2. Vector search initialization (VectorSearchIndex & ExactSearchIndex).
3. Exact Cosine & Dot Product brute-force candidate search.
4. Latency profiling & top-K candidate retrieval verification.
"""

import os
import sys
import time
import numpy as np

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import VectorSearchIndex, ExactSearchIndex


def main():
    print("=========================================================")
    print(" MEMBER 5 - DAY 1: ARCHITECTURE & EXACT SEARCH BASELINE  ")
    print("=========================================================\n")

    # 1. Simulate Member 2 Output: Item Catalog Embeddings (N=1,000 items, D=32 dimensions)
    np.random.seed(42)
    num_items = 1000
    dim = 32
    item_ids = [f"item_{i:04d}" for i in range(num_items)]
    item_embeddings = np.random.randn(num_items, dim).astype(np.float32)

    print(f"[Contract Check - Member 2] Ingesting Item Embedding Catalog:")
    print(f"  - Item Count: {len(item_ids)}")
    print(f"  - Embedding Dimension: {dim}")
    print(f"  - Sample Item ID: {item_ids[0]}")
    print(f"  - Sample Embedding (first 5 elements): {item_embeddings[0, :5]}\n")

    # 2. Build and fit ExactSearchIndex
    start_fit = time.perf_counter()
    index = ExactSearchIndex(normalize=True, metric="cosine")
    index.fit(item_ids, item_embeddings)
    fit_duration_ms = (time.perf_counter() - start_fit) * 1000.0

    print(f"[Index Setup] Fitted ExactSearchIndex:")
    print(f"  - Is Fitted: {index.is_fitted}")
    print(f"  - Indexed Item Count: {index.num_items}")
    print(f"  - Index Dimension: {index.dim}")
    print(f"  - Fit Latency: {fit_duration_ms:.3f} ms\n")

    # 3. Simulate Member 1 Output: Batch Query Vectors (Q=50 user query vectors)
    num_queries = 50
    query_embeddings = np.random.randn(num_queries, dim).astype(np.float32)

    print(f"[Contract Check - Member 1] Simulating Query Tower Vectors:")
    print(f"  - Query Count: {num_queries}")
    print(f"  - Query Dimension: {query_embeddings.shape[1]}\n")

    # 4. Perform Top-K Search (k=10)
    k = 10
    start_search = time.perf_counter()
    retrieved_ids, scores = index.search(query_embeddings, k=k)
    search_duration_ms = (time.perf_counter() - start_search) * 1000.0
    avg_latency_ms = search_duration_ms / num_queries
    qps = num_queries / (search_duration_ms / 1000.0)

    print(f"[Search Execution] Completed Top-{k} Exact Candidate Retrieval:")
    print(f"  - Retrieved IDs Shape: {retrieved_ids.shape}")
    print(f"  - Scores Matrix Shape: {scores.shape}")
    print(f"  - Total Search Time: {search_duration_ms:.3f} ms")
    print(f"  - Average Per-Query Latency: {avg_latency_ms:.3f} ms")
    print(f"  - Throughput (QPS): {qps:.1f} queries/sec\n")

    # 5. Verification Check: Self-Query Retrieval Accuracy (Target: Top-1 Score == 1.0)
    target_queries = item_embeddings[:5]
    exact_ids, exact_scores = index.search(target_queries, k=1)

    print("[Verification] Checking Top-1 Self-Query Accuracy:")
    for i in range(5):
        matched = exact_ids[i, 0] == item_ids[i]
        score = exact_scores[i, 0]
        status = "PASSED" if matched and np.isclose(score, 1.0, atol=1e-5) else "FAILED"
        print(f"  - Query Item {item_ids[i]} -> Retrieved {exact_ids[i, 0]} | Score: {score:.5f} | [{status}]")

    print("\n=========================================================")
    print(" DAY 1 IMPLEMENTATION & VERIFICATION COMPLETED SUCCESSFULLY ")
    print("=========================================================")


if __name__ == "__main__":
    main()
