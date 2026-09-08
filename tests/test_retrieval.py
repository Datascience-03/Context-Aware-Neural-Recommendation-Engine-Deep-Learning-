"""
Unit tests for Day 4: Vector Retrieval & Approximate Nearest Neighbor (ANN) Indexing.
Tests ExactSearchIndex, IVFIndex, and ANNEvaluator trade-off benchmarking.
"""

import pytest
import numpy as np
import pandas as pd

from src.retrieval import (
    VectorSearchIndex,
    ExactSearchIndex,
    IVFIndex,
    ANNEvaluator,
)


@pytest.fixture
def synthetic_catalog():
    """Generate reproducible synthetic item embeddings and query vectors."""
    np.random.seed(42)
    num_items = 500
    dim = 32
    item_ids = [f"item_{i:03d}" for i in range(num_items)]
    item_embeddings = np.random.randn(num_items, dim).astype(np.float32)

    num_queries = 25
    query_embeddings = np.random.randn(num_queries, dim).astype(np.float32)

    return {
        "item_ids": item_ids,
        "item_embeddings": item_embeddings,
        "query_embeddings": query_embeddings,
        "dim": dim,
        "num_items": num_items,
        "num_queries": num_queries,
    }


def test_exact_search_index_accuracy(synthetic_catalog):
    """Test exact search retrieves the identical vector with similarity 1.0."""
    item_ids = synthetic_catalog["item_ids"]
    embeddings = synthetic_catalog["item_embeddings"]

    index = ExactSearchIndex(normalize=True)
    index.fit(item_ids, embeddings)

    assert index.is_fitted
    assert index.num_items == len(item_ids)
    assert index.dim == synthetic_catalog["dim"]

    # Query with exact first 5 items from the catalog
    target_queries = embeddings[:5]
    retrieved_ids, scores = index.search(target_queries, k=3)

    assert retrieved_ids.shape == (5, 3)
    assert scores.shape == (5, 3)

    # Top-1 retrieved ID must be identical to the query item
    for i in range(5):
        assert retrieved_ids[i, 0] == item_ids[i]
        assert np.isclose(scores[i, 0], 1.0, atol=1e-5)


def test_exact_search_k_greater_than_catalog(synthetic_catalog):
    """Test search handles k >= num_items cleanly without index errors."""
    item_ids = synthetic_catalog["item_ids"][:10]
    embeddings = synthetic_catalog["item_embeddings"][:10]

    index = ExactSearchIndex(normalize=True)
    index.fit(item_ids, embeddings)

    query = synthetic_catalog["query_embeddings"][:2]
    retrieved_ids, scores = index.search(query, k=50)

    assert retrieved_ids.shape == (2, 10)
    assert scores.shape == (2, 10)


def test_unfitted_index_raises():
    """Unfitted index should raise RuntimeError on search."""
    index = ExactSearchIndex()
    with pytest.raises(RuntimeError):
        index.search(np.random.randn(2, 16))


def test_dimension_mismatch_raises(synthetic_catalog):
    """Dimension mismatch between query and index must raise ValueError."""
    index = ExactSearchIndex()
    index.fit(synthetic_catalog["item_ids"], synthetic_catalog["item_embeddings"])

    wrong_dim_query = np.random.randn(2, 64)  # dim is 32
    with pytest.raises(ValueError, match="does not match index dimension"):
        index.search(wrong_dim_query, k=5)


def test_ivf_index_construction_and_search(synthetic_catalog):
    """Test IVF index clustering, inverted lists, and candidate retrieval."""
    item_ids = synthetic_catalog["item_ids"]
    embeddings = synthetic_catalog["item_embeddings"]
    queries = synthetic_catalog["query_embeddings"]

    nlist = 16
    ivf = IVFIndex(nlist=nlist, nprobe=2, normalize=True, random_state=42)
    ivf.fit(item_ids, embeddings)

    assert ivf.is_fitted
    assert ivf.centroids.shape == (nlist, synthetic_catalog["dim"])
    assert len(ivf.inverted_lists) == nlist

    # Check total items stored in inverted lists equals total indexed items
    total_indexed = sum(len(lst) for lst in ivf.inverted_lists.values())
    assert total_indexed == len(item_ids)

    # Search top-5
    k = 5
    retrieved_ids, scores = ivf.search(queries, k=k)
    assert retrieved_ids.shape == (len(queries), k)
    assert scores.shape == (len(queries), k)

    # All retrieved IDs must be valid items from catalog
    valid_set = set(item_ids)
    for q_ids in retrieved_ids:
        for it in q_ids:
            if it is not None:
                assert it in valid_set


def test_ivf_full_probe_matches_exact(synthetic_catalog):
    """When nprobe == nlist, IVF probes all cells and matches exact search 100%."""
    item_ids = synthetic_catalog["item_ids"]
    embeddings = synthetic_catalog["item_embeddings"]
    queries = synthetic_catalog["query_embeddings"][:10]

    exact_index = ExactSearchIndex(normalize=True)
    exact_index.fit(item_ids, embeddings)
    exact_ids, _ = exact_index.search(queries, k=5)

    nlist = 8
    ivf = IVFIndex(nlist=nlist, nprobe=nlist, normalize=True, random_state=42)
    ivf.fit(item_ids, embeddings)
    ivf_ids, _ = ivf.search(queries, k=5, nprobe=nlist)

    retention = ANNEvaluator.recall_retention_at_k(exact_ids, ivf_ids, k=5)
    assert np.isclose(retention, 1.0, atol=1e-5)


def test_ann_evaluator_metrics():
    """Test ANNEvaluator calculation of recall retention and top-1 accuracy."""
    exact = np.array([
        ["item_1", "item_2", "item_3"],
        ["item_4", "item_5", "item_6"],
    ])

    # Case 1: 100% retention
    ann_perfect = np.array([
        ["item_1", "item_2", "item_3"],
        ["item_4", "item_5", "item_6"],
    ])
    assert ANNEvaluator.recall_retention_at_k(exact, ann_perfect, k=3) == 1.0
    assert ANNEvaluator.top_1_accuracy(exact, ann_perfect) == 1.0

    # Case 2: Partial overlap (2 out of 3 for query 0, 1 out of 3 for query 1)
    ann_partial = np.array([
        ["item_1", "item_2", "item_99"],  # 2/3
        ["item_4", "item_88", "item_99"],  # 1/3
    ])
    expected_recall = (2/3 + 1/3) / 2.0  # 0.5
    assert np.isclose(ANNEvaluator.recall_retention_at_k(exact, ann_partial, k=3), expected_recall)
    # Both top-1 match
    assert ANNEvaluator.top_1_accuracy(exact, ann_partial) == 1.0

    # Case 3: Zero overlap
    ann_zero = np.array([
        ["item_77", "item_88", "item_99"],
        ["item_77", "item_88", "item_99"],
    ])
    assert ANNEvaluator.recall_retention_at_k(exact, ann_zero, k=3) == 0.0
    assert ANNEvaluator.top_1_accuracy(exact, ann_zero) == 0.0


def test_ann_evaluator_single_evaluation(synthetic_catalog):
    """Test full single-run evaluation returning expected dictionary metrics."""
    item_ids = synthetic_catalog["item_ids"]
    embeddings = synthetic_catalog["item_embeddings"]
    queries = synthetic_catalog["query_embeddings"]

    exact_index = ExactSearchIndex(normalize=True).fit(item_ids, embeddings)
    ivf_index = IVFIndex(nlist=8, nprobe=2, normalize=True, random_state=42).fit(item_ids, embeddings)

    metrics = ANNEvaluator.evaluate_ann_retrieval(
        ann_index=ivf_index,
        exact_index=exact_index,
        queries=queries,
        k=5,
        benchmark_runs=2,
    )

    assert "recall_retention@5" in metrics
    assert "top1_accuracy" in metrics
    assert "ann_mean_latency_ms" in metrics
    assert "ann_qps" in metrics
    assert "speedup_factor" in metrics
    assert 0.0 <= metrics["recall_retention@5"] <= 1.0
    assert metrics["ann_qps"] > 0


def test_benchmark_tradeoff_sweep(synthetic_catalog):
    """Test Pareto trade-off sweep produces a sorted DataFrame with monotonic recall trends."""
    item_ids = synthetic_catalog["item_ids"]
    embeddings = synthetic_catalog["item_embeddings"]
    queries = synthetic_catalog["query_embeddings"]

    exact_index = ExactSearchIndex(normalize=True).fit(item_ids, embeddings)
    ivf_index = IVFIndex(nlist=16, nprobe=1, normalize=True, random_state=42).fit(item_ids, embeddings)

    probes = [1, 2, 4, 8, 16]
    df = ANNEvaluator.benchmark_tradeoff_sweep(
        ann_index=ivf_index,
        exact_index=exact_index,
        queries=queries,
        k=5,
        nprobe_list=probes,
        benchmark_runs=2,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(probes)
    assert list(df["nprobe"]) == probes

    # Check required columns
    expected_cols = ["nprobe", "nlist", "recall_retention@5", "top1_accuracy", "latency_ms", "qps", "speedup_vs_exact"]
    for col in expected_cols:
        assert col in df.columns

    # Check recall retention increases as nprobe increases
    recalls = df["recall_retention@5"].tolist()
    assert recalls[0] <= recalls[-1]
    assert np.isclose(recalls[-1], 1.0, atol=1e-5)
