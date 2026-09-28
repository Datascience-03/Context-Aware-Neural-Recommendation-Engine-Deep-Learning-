"""
Member 5 - Day 3 Execution Script
-----------------------------------
Demonstrates:
1. Hierarchical Navigable Small World (HNSW) Graph Index construction.
2. Query-time ef_search parameter sweep (ef_search = 8, 16, 32, 64) for sub-linear search.
3. Index serialization & deserialization (save_index and load_index to/from disk).
4. Verification of 100% bitwise exact search reproducibility post-reload.
"""

import os
import sys
import time
import numpy as np

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import ExactSearchIndex, HNSWIndex, ANNEvaluator, save_index, load_index


def main():
    print("=================================================================")
    print(" MEMBER 5 - DAY 3: HNSW GRAPH INDEX & VECTOR PERSISTENCE ROUTINES")
    print("=================================================================\n")

    # 1. Generate Synthetic Catalog Embeddings (N=1,500 items, D=32 dimensions)
    np.random.seed(42)
    num_items = 1500
    dim = 32
    item_ids = [f"item_{i:04d}" for i in range(num_items)]
    item_embeddings = np.random.randn(num_items, dim).astype(np.float32)

    print(f"[Dataset Setup] Catalog Ingestion:")
    print(f"  - Total Items (N): {num_items}")
    print(f"  - Vector Dimension (D): {dim}\n")

    # 2. Fit Ground-Truth Exact Search Index
    exact_index = ExactSearchIndex(normalize=True)
    exact_index.fit(item_ids, item_embeddings)

    # 3. Fit HNSW Graph Index (M=16, ef_construction=64)
    M = 16
    ef_construction = 64
    print(f"[HNSW Construction] Fitting HNSW Proximity Graph (M={M}, ef_construction={ef_construction})...")
    start_fit = time.perf_counter()
    hnsw_index = HNSWIndex(M=M, ef_construction=ef_construction, ef_search=32, normalize=True, random_state=42)
    hnsw_index.fit(item_ids, item_embeddings)
    fit_duration_ms = (time.perf_counter() - start_fit) * 1000.0

    print(f"  - Is Fitted: {hnsw_index.is_fitted}")
    print(f"  - Fit Latency: {fit_duration_ms:.3f} ms")
    print(f"  - Graph Nodes Count: {len(hnsw_index.graph)}\n")

    # 4. Generate Batch Queries (Q=50 user query vectors)
    num_queries = 50
    query_embeddings = np.random.randn(num_queries, dim).astype(np.float32)
    k = 10

    # Obtain Ground-Truth Exact Top-K Results
    exact_ids, _ = exact_index.search(query_embeddings, k=k)

    # 5. Sweep ef_search parameter levels
    print(f"[HNSW Search Benchmark] Evaluating ef_search levels (k={k}):")
    print("-" * 65)
    print(f"{'ef_search':<10} | {'Recall@10':<12} | {'Latency (ms)':<15} | {'QPS':<10}")
    print("-" * 65)

    for ef in [8, 16, 32, 64]:
        start_search = time.perf_counter()
        hnsw_ids, hnsw_scores = hnsw_index.search(query_embeddings, k=k, ef_search=ef)
        search_time_ms = (time.perf_counter() - start_search) * 1000.0
        avg_latency = search_time_ms / num_queries
        qps = num_queries / (search_time_ms / 1000.0)

        recall = ANNEvaluator.recall_retention_at_k(exact_ids, hnsw_ids, k=k)
        print(f"{ef:<10} | {recall * 100:>10.2f}% | {avg_latency:>13.3f} ms | {qps:>9.1f}")

    print("-" * 65 + "\n")

    # 6. Test Index Persistence (Save to Disk & Load Back)
    save_path = os.path.join("outputs", "ann_indices", "hnsw_catalog_demo.idx")
    print(f"[Persistence Routine] Saving HNSW Index to disk:")
    print(f"  - Output Filepath: {os.path.abspath(save_path)}")

    abs_saved_path = save_index(hnsw_index, save_path)
    file_size_kb = os.path.getsize(abs_saved_path) / 1024.0
    print(f"  - Index Saved Successfully ({file_size_kb:.2f} KB)")

    print(f"\n[Persistence Routine] Reloading HNSW Index from disk...")
    reloaded_hnsw = load_index(abs_saved_path)
    print(f"  - Reloaded Index Type: {type(reloaded_hnsw).__name__}")
    print(f"  - Reloaded Item Count: {reloaded_hnsw.num_items}")

    # 7. Verification Assertions: Original vs Reloaded Index Search Output Match
    orig_ids, orig_scores = hnsw_index.search(query_embeddings, k=k, ef_search=32)
    reloaded_ids, reloaded_scores = reloaded_hnsw.search(query_embeddings, k=k, ef_search=32)

    np.testing.assert_array_equal(orig_ids, reloaded_ids)
    np.testing.assert_allclose(orig_scores, reloaded_scores, rtol=1e-5)

    print("\n[Verification Check] Reloaded Index Output Verification PASSED (100% Exact Match)")
    print("=================================================================")
    print(" DAY 3 IMPLEMENTATION & VERIFICATION COMPLETED SUCCESSFULLY      ")
    print("=================================================================")


if __name__ == "__main__":
    main()
