"""
Member 5 - Day 2 Execution Script
-----------------------------------
Demonstrates:
1. Inverted File (IVF) Index construction with spherical K-Means Voronoi partitioning.
2. Two-stage search algorithm (Centroid Selection + Inverted List Sub-search).
3. Multi-cell probing trade-off evaluation (nprobe=1 vs nprobe=3 vs nprobe=5).
4. Recall Retention Rate (@K=10) & latency/QPS metrics benchmarked against exact ground truth.
"""

import os
import sys
import time
import numpy as np

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import ExactSearchIndex, IVFIndex, ANNEvaluator


def main():
    print("=================================================================")
    print(" MEMBER 5 - DAY 2: IVF INDEX & K-MEANS VORONOI PARTITIONING     ")
    print("=================================================================\n")

    # 1. Generate Synthetic Catalog Embeddings (N=2,000 items, D=32 dimensions)
    np.random.seed(42)
    num_items = 2000
    dim = 32
    item_ids = [f"item_{i:04d}" for i in range(num_items)]
    item_embeddings = np.random.randn(num_items, dim).astype(np.float32)

    print(f"[Dataset Setup] Catalog Ingestion:")
    print(f"  - Total Items (N): {num_items}")
    print(f"  - Vector Dimension (D): {dim}\n")

    # 2. Fit Ground-Truth Exact Search Index
    exact_index = ExactSearchIndex(normalize=True)
    exact_index.fit(item_ids, item_embeddings)

    # 3. Fit IVF Index (nlist=20 clusters)
    nlist = 20
    print(f"[IVF Index Construction] Fitting IVF Index with nlist={nlist} Voronoi clusters...")
    start_fit = time.perf_counter()
    ivf_index = IVFIndex(nlist=nlist, nprobe=1, normalize=True, max_iter=30, random_state=42)
    ivf_index.fit(item_ids, item_embeddings)
    fit_duration_ms = (time.perf_counter() - start_fit) * 1000.0

    print(f"  - Is Fitted: {ivf_index.is_fitted}")
    print(f"  - Fit Latency: {fit_duration_ms:.3f} ms")
    print(f"  - Centroids Shape: {ivf_index.centroids.shape}")

    # Inspect Inverted List Partition Distribution
    cluster_sizes = [len(items) for items in ivf_index.inverted_lists.values()]
    print(f"  - Inverted List Sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}\n")

    # 4. Generate Batch Queries (Q=100 user query vectors)
    num_queries = 100
    query_embeddings = np.random.randn(num_queries, dim).astype(np.float32)
    k = 10

    # Obtain Ground-Truth Exact Top-K Results
    exact_ids, exact_scores = exact_index.search(query_embeddings, k=k)

    # 5. Evaluate Multi-Cell Probing (nprobe = 1, 3, 5, 10)
    print(f"[Search Execution & Probing Benchmark] Comparing nprobe levels (k={k}):")
    print("-" * 65)
    print(f"{'nprobe':<8} | {'Recall@10':<12} | {'Latency (ms)':<15} | {'QPS':<10}")
    print("-" * 65)

    for nprobe in [1, 3, 5, 10]:
        start_search = time.perf_counter()
        ivf_ids, ivf_scores = ivf_index.search(query_embeddings, k=k, nprobe=nprobe)
        search_time_ms = (time.perf_counter() - start_search) * 1000.0
        avg_latency = search_time_ms / num_queries
        qps = num_queries / (search_time_ms / 1000.0)

        # Calculate Recall Retention against Exact Brute-Force
        recall = ANNEvaluator.recall_retention_at_k(exact_ids, ivf_ids, k=k)

        print(f"{nprobe:<8} | {recall * 100:>10.2f}% | {avg_latency:>13.3f} ms | {qps:>9.1f}")

    print("-" * 65)

    # 6. Verification Assertions
    # Verify higher nprobe achieves higher or equal recall retention
    ivf_ids_p1, _ = ivf_index.search(query_embeddings, k=k, nprobe=1)
    ivf_ids_p10, _ = ivf_index.search(query_embeddings, k=k, nprobe=10)
    recall_p1 = ANNEvaluator.recall_retention_at_k(exact_ids, ivf_ids_p1, k=k)
    recall_p10 = ANNEvaluator.recall_retention_at_k(exact_ids, ivf_ids_p10, k=k)

    assert recall_p10 >= recall_p1, "Higher nprobe should yield equal or higher recall"
    print("\n[Verification Check] Probing Recall Monotonicity PASSED (Recall@10 nprobe=10 >= nprobe=1)")
    print("=================================================================")
    print(" DAY 2 IMPLEMENTATION & VERIFICATION COMPLETED SUCCESSFULLY      ")
    print("=================================================================")


if __name__ == "__main__":
    main()
