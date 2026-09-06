"""
Evaluation Module for Context-Aware Neural Recommendation Engine.
Provides ranking metrics, evaluation dataset preparation, and retrieval evaluator pipeline.
"""

from src.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    mrr_at_k,
    hit_rate_at_k,
    evaluate_batch_metrics,
)
from src.evaluation.evaluator import (
    EvaluationDataset,
    RetrievalEvaluator,
)

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "hit_rate_at_k",
    "evaluate_batch_metrics",
    "EvaluationDataset",
    "RetrievalEvaluator",
]
