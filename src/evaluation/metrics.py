"""
Evaluation metrics for Context-Aware Neural Recommendation Engine.
Provides Recall@K, NDCG@K, MRR@K, Hit Rate@K, and batch metric calculation utilities.
"""

import math
from typing import Dict, List, Set, Union, Sequence


def recall_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Calculate Recall@K for a single user/query session.

    Recall@K = |actual ∩ predicted[:k]| / |actual|

    Args:
        actual: List of ground-truth relevant item IDs.
        predicted: Ordered list of recommended item IDs (ranked best to worst).
        k: Top-K cutoff.

    Returns:
        float: Recall score in range [0.0, 1.0].
    """
    if not actual or k <= 0:
        return 0.0

    actual_set: Set[Union[int, str]] = set(actual)
    top_k_predicted = predicted[:k]

    hits = sum(1 for item in top_k_predicted if item in actual_set)
    return hits / len(actual_set)


def hit_rate_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Calculate Hit Rate@K (1.0 if at least 1 relevant item is retrieved in top-K, else 0.0).

    Args:
        actual: List of ground-truth relevant item IDs.
        predicted: Ordered list of recommended item IDs.
        k: Top-K cutoff.

    Returns:
        float: 1.0 or 0.0.
    """
    if not actual or k <= 0:
        return 0.0

    actual_set: Set[Union[int, str]] = set(actual)
    top_k_predicted = predicted[:k]

    for item in top_k_predicted:
        if item in actual_set:
            return 1.0
    return 0.0


def mrr_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR@K) for a single user/query session.

    Reciprocal Rank of the first relevant item in the top-K recommendations.

    Args:
        actual: List of ground-truth relevant item IDs.
        predicted: Ordered list of recommended item IDs.
        k: Top-K cutoff.

    Returns:
        float: Reciprocal rank score in range [0.0, 1.0].
    """
    if not actual or k <= 0:
        return 0.0

    actual_set: Set[Union[int, str]] = set(actual)
    top_k_predicted = predicted[:k]

    for rank_idx, item in enumerate(top_k_predicted, start=1):
        if item in actual_set:
            return 1.0 / rank_idx
    return 0.0


def ndcg_at_k(actual: Sequence[Union[int, str]], predicted: Sequence[Union[int, str]], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K) with binary relevance.

    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1))
    IDCG@K = sum_{i=1}^{min(|actual|, K)} (1 / log2(i + 1))
    NDCG@K = DCG@K / IDCG@K

    Args:
        actual: List of ground-truth relevant item IDs.
        predicted: Ordered list of recommended item IDs.
        k: Top-K cutoff.

    Returns:
        float: NDCG score in range [0.0, 1.0].
    """
    if not actual or k <= 0:
        return 0.0

    actual_set: Set[Union[int, str]] = set(actual)
    top_k_predicted = predicted[:k]

    dcg = 0.0
    for rank_idx, item in enumerate(top_k_predicted, start=1):
        if item in actual_set:
            dcg += 1.0 / math.log2(rank_idx + 1)

    # Ideal DCG assumes all actual relevant items appear at top positions
    ideal_hits = min(len(actual_set), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank_idx + 1) for rank_idx in range(1, ideal_hits + 1))

    return dcg / idcg


def evaluate_batch_metrics(
    actual_list: List[Sequence[Union[int, str]]],
    predicted_list: List[Sequence[Union[int, str]]],
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Calculate average metrics across a batch of user query sessions.

    Args:
        actual_list: List of ground-truth relevant item lists (one list per session).
        predicted_list: List of predicted top item recommendations (one list per session).
        k_list: List of K thresholds to evaluate.

    Returns:
        Dict[str, float]: Dictionary mapping metric names (e.g. 'recall@10', 'ndcg@10') to mean values.
    """
    if len(actual_list) != len(predicted_list):
        raise ValueError(
            f"Length mismatch: actual_list ({len(actual_list)}) vs predicted_list ({len(predicted_list)})"
        )

    num_sessions = len(actual_list)
    if num_sessions == 0:
        results = {}
        for k in k_list:
            results[f"recall@{k}"] = 0.0
            results[f"ndcg@{k}"] = 0.0
            results[f"mrr@{k}"] = 0.0
            results[f"hit_rate@{k}"] = 0.0
        return results

    results = {}
    for k in k_list:
        tot_recall = sum(recall_at_k(act, pred, k) for act, pred in zip(actual_list, predicted_list))
        tot_ndcg = sum(ndcg_at_k(act, pred, k) for act, pred in zip(actual_list, predicted_list))
        tot_mrr = sum(mrr_at_k(act, pred, k) for act, pred in zip(actual_list, predicted_list))
        tot_hr = sum(hit_rate_at_k(act, pred, k) for act, pred in zip(actual_list, predicted_list))

        results[f"recall@{k}"] = round(tot_recall / num_sessions, 5)
        results[f"ndcg@{k}"] = round(tot_ndcg / num_sessions, 5)
        results[f"mrr@{k}"] = round(tot_mrr / num_sessions, 5)
        results[f"hit_rate@{k}"] = round(tot_hr / num_sessions, 5)

    return results
