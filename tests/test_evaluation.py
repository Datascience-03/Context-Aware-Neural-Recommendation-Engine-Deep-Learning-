"""
Unit tests for Recommendation Evaluation Metrics.
"""

import pytest
import math
from src.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    mrr_at_k,
    hit_rate_at_k,
    evaluate_batch_metrics,
)


def test_recall_at_k_perfect_and_zero():
    actual = [10, 20, 30]
    predicted_perfect = [10, 20, 30, 40, 50]
    predicted_zero = [40, 50, 60]

    assert recall_at_k(actual, predicted_perfect, k=3) == 1.0
    assert recall_at_k(actual, predicted_perfect, k=2) == 2 / 3
    assert recall_at_k(actual, predicted_zero, k=5) == 0.0


def test_recall_at_k_empty_edge_cases():
    assert recall_at_k([], [1, 2, 3], k=5) == 0.0
    assert recall_at_k([1, 2], [], k=5) == 0.0
    assert recall_at_k([1, 2], [1, 2], k=0) == 0.0
    assert recall_at_k([1, 2], [1, 2], k=-1) == 0.0


def test_hit_rate_at_k():
    actual = ["item_a", "item_b"]
    predicted_hit = ["item_x", "item_a", "item_y"]
    predicted_miss = ["item_x", "item_y", "item_z"]

    assert hit_rate_at_k(actual, predicted_hit, k=2) == 1.0
    assert hit_rate_at_k(actual, predicted_hit, k=1) == 0.0
    assert hit_rate_at_k(actual, predicted_miss, k=3) == 0.0


def test_mrr_at_k():
    actual = [100]
    predicted_rank_1 = [100, 200, 300]
    predicted_rank_2 = [200, 100, 300]
    predicted_rank_3 = [200, 300, 100]
    predicted_none = [200, 300, 400]

    assert mrr_at_k(actual, predicted_rank_1, k=3) == 1.0
    assert mrr_at_k(actual, predicted_rank_2, k=3) == 0.5
    assert mrr_at_k(actual, predicted_rank_3, k=3) == pytest.approx(1 / 3)
    assert mrr_at_k(actual, predicted_rank_3, k=2) == 0.0  # Cutoff before rank 3
    assert mrr_at_k(actual, predicted_none, k=3) == 0.0


def test_ndcg_at_k_ranking_sensitivity():
    actual = [1, 2]
    # Prediction A: Relevant items at position 1 and 2 (perfect)
    pred_perfect = [1, 2, 3, 4]
    # Prediction B: Relevant items at position 2 and 3
    pred_later = [3, 1, 2, 4]

    ndcg_perfect = ndcg_at_k(actual, pred_perfect, k=4)
    ndcg_later = ndcg_at_k(actual, pred_later, k=4)

    assert ndcg_perfect == 1.0
    assert 0.0 < ndcg_later < ndcg_perfect

    # Manually check DCG/IDCG for pred_later
    # Rank 2 (item 1): 1 / log2(3)
    # Rank 3 (item 2): 1 / log2(4)
    # IDCG: 1/log2(2) + 1/log2(3)
    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    expected_ndcg = expected_dcg / expected_idcg

    assert ndcg_later == pytest.approx(expected_ndcg)


def test_evaluate_batch_metrics():
    actual_list = [
        [10, 20],
        [30],
        [40, 50, 60]
    ]
    predicted_list = [
        [10, 99, 20],   # User 1: recall@3 = 1.0, hr@3 = 1.0
        [99, 30, 98],   # User 2: recall@3 = 1.0, hr@3 = 1.0
        [99, 98, 97]    # User 3: recall@3 = 0.0, hr@3 = 0.0
    ]

    metrics = evaluate_batch_metrics(actual_list, predicted_list, k_list=[3])

    assert "recall@3" in metrics
    assert "ndcg@3" in metrics
    assert "mrr@3" in metrics
    assert "hit_rate@3" in metrics

    # Recall mean = (1.0 + 1.0 + 0.0) / 3 = 0.66667
    assert metrics["recall@3"] == pytest.approx(2 / 3, abs=1e-4)
    # Hit Rate mean = (1.0 + 1.0 + 0.0) / 3 = 0.66667
    assert metrics["hit_rate@3"] == pytest.approx(2 / 3, abs=1e-4)


def test_evaluate_batch_metrics_mismatch():
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_batch_metrics([[1]], [[1], [2]], k_list=[5])
