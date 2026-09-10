"""
Evaluation Module for Context-Aware Neural Recommendation Engine.
Provides ranking metrics, evaluation dataset preparation, retrieval evaluator pipeline,
baseline recommendation models with comparative benchmarking harness,
and context-slice & beyond-accuracy diversity/coverage evaluation.
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
from src.evaluation.baselines import (
    BaseRecommender,
    PopularityRecommender,
    RecentPopularityRecommender,
    UserHistoryRecommender,
    RandomRecommender,
    BenchmarkHarness,
)
from src.evaluation.slice_evaluator import (
    catalog_coverage,
    intra_list_diversity,
    prediction_entropy,
    gini_coefficient,
    novelty_at_k,
    SliceEvaluator,
)

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "hit_rate_at_k",
    "evaluate_batch_metrics",
    "EvaluationDataset",
    "RetrievalEvaluator",
    "BaseRecommender",
    "PopularityRecommender",
    "RecentPopularityRecommender",
    "UserHistoryRecommender",
    "RandomRecommender",
    "BenchmarkHarness",
    "catalog_coverage",
    "intra_list_diversity",
    "prediction_entropy",
    "gini_coefficient",
    "novelty_at_k",
    "SliceEvaluator",
]
