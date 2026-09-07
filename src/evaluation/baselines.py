"""
Day 3: Baseline Recommendation Models & Comparative Benchmarking Harness.
Provides standard heuristic and statistical baselines for recommendation benchmarking:
- PopularityRecommender: Global frequency ranker.
- RecentPopularityRecommender: Time-decayed recency-weighted popularity.
- UserHistoryRecommender: Prior user repeat-interaction baseline with popularity fallback.
- RandomRecommender: Uniform random sampling (theoretical lower bound).
- BenchmarkHarness: Comparative evaluation suite for benchmarking multiple models.
"""

import os
import sys
import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

# Append workspace root to path for standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    mrr_at_k,
    hit_rate_at_k,
    evaluate_batch_metrics,
)
from src.evaluation.evaluator import EvaluationDataset, RetrievalEvaluator

logger = logging.getLogger(__name__)


class BaseRecommender(abc.ABC):
    """Abstract base class for recommendation models and baselines."""

    @abc.abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseRecommender":
        """Fits the recommender on training interaction records."""
        raise NotImplementedError

    @abc.abstractmethod
    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_pool: Optional[List[str]] = None,
    ) -> List[str]:
        """Generates top-K recommended item IDs for a user."""
        raise NotImplementedError

    def predict_batch(
        self,
        user_ids: Sequence[str],
        k: int = 10,
        candidate_pools: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generates top-K recommendations for a batch of users.

        Args:
            user_ids: Sequence of user IDs to recommend for.
            k: Number of recommendations per user.
            candidate_pools: Optional candidate pool dictionary per user.

        Returns:
            Dict[str, List[str]]: Mapping from user_id to top-K recommended item IDs.
        """
        predictions: Dict[str, List[str]] = {}
        for uid in user_ids:
            pool = candidate_pools.get(str(uid)) if candidate_pools else None
            predictions[str(uid)] = self.recommend(str(uid), k=k, candidate_pool=pool)
        return predictions


class PopularityRecommender(BaseRecommender):
    """
    Most Popular Items Recommender.
    Ranks items based on global historical interaction frequencies.
    A crucial standard baseline that all deep learning models must surpass.
    """

    def __init__(self, user_col: str = "customer_id", item_col: str = "article_id"):
        self.user_col = user_col
        self.item_col = item_col
        self.item_counts: pd.Series = pd.Series(dtype=int)
        self.top_items: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        clean_df = train_df.dropna(subset=[self.item_col]).copy()
        clean_df[self.item_col] = clean_df[self.item_col].astype(str).str.strip()

        # Compute frequency counts
        self.item_counts = clean_df[self.item_col].value_counts()
        self.top_items = self.item_counts.index.tolist()
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_pool: Optional[List[str]] = None,
    ) -> List[str]:
        if candidate_pool is not None:
            # Rank the candidate pool by popularity frequency
            ranked = sorted(
                candidate_pool,
                key=lambda x: self.item_counts.get(str(x), 0),
                reverse=True,
            )
            return [str(x) for x in ranked[:k]]
        return [str(x) for x in self.top_items[:k]]


class RecentPopularityRecommender(BaseRecommender):
    """
    Recency-Weighted Popularity Recommender.
    Applies exponential time-decay weighting to interactions so recent purchases
    have higher impact than distant past purchases:
        weight = exp(-decay_rate * days_ago)
    Captures temporal trends and seasonal shifts.
    """

    def __init__(
        self,
        user_col: str = "customer_id",
        item_col: str = "article_id",
        time_col: str = "t_dat",
        half_life_days: float = 14.0,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.half_life_days = max(half_life_days, 1.0)
        self.decay_rate = np.log(2) / self.half_life_days
        self.item_scores: Dict[str, float] = {}
        self.top_items: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "RecentPopularityRecommender":
        clean_df = train_df.dropna(subset=[self.item_col, self.time_col]).copy()
        clean_df[self.item_col] = clean_df[self.item_col].astype(str).str.strip()

        # Parse timestamps to dates
        clean_df["_dt"] = pd.to_datetime(clean_df[self.time_col], errors="coerce")
        max_dt = clean_df["_dt"].max()

        if pd.isna(max_dt):
            # Fallback to simple frequency if date parsing fails
            clean_df["weight"] = 1.0
        else:
            days_ago = (max_dt - clean_df["_dt"]).dt.total_seconds() / (24 * 3600)
            days_ago = np.maximum(days_ago.fillna(0), 0)
            clean_df["weight"] = np.exp(-self.decay_rate * days_ago)

        weighted_counts = clean_df.groupby(self.item_col)["weight"].sum().sort_values(ascending=False)
        self.item_scores = weighted_counts.to_dict()
        self.top_items = weighted_counts.index.tolist()
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_pool: Optional[List[str]] = None,
    ) -> List[str]:
        if candidate_pool is not None:
            ranked = sorted(
                candidate_pool,
                key=lambda x: self.item_scores.get(str(x), 0.0),
                reverse=True,
            )
            return [str(x) for x in ranked[:k]]
        return [str(x) for x in self.top_items[:k]]


class UserHistoryRecommender(BaseRecommender):
    """
    User Repeat Interaction Recommender.
    Recommends items the user has previously bought or clicked (frequently repurchased items),
    falling back to global popularity when the user has fewer than K past items.
    """

    def __init__(
        self,
        user_col: str = "customer_id",
        item_col: str = "article_id",
        time_col: Optional[str] = "t_dat",
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.user_history: Dict[str, List[str]] = {}
        self.global_popularity: PopularityRecommender = PopularityRecommender(user_col, item_col)

    def fit(self, train_df: pd.DataFrame) -> "UserHistoryRecommender":
        clean_df = train_df.dropna(subset=[self.user_col, self.item_col]).copy()
        clean_df[self.user_col] = clean_df[self.user_col].astype(str).str.strip()
        clean_df[self.item_col] = clean_df[self.item_col].astype(str).str.strip()

        # Fit fallback popularity
        self.global_popularity.fit(clean_df)

        if self.time_col and self.time_col in clean_df.columns:
            sorted_df = clean_df.sort_values(by=[self.user_col, self.time_col], ascending=[True, False])
        else:
            sorted_df = clean_df

        # Store ordered unique items per user (most recent first if time_col provided)
        self.user_history = (
            sorted_df.groupby(self.user_col)[self.item_col]
            .apply(lambda s: list(dict.fromkeys(s)))
            .to_dict()
        )
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_pool: Optional[List[str]] = None,
    ) -> List[str]:
        past_items = self.user_history.get(str(user_id), [])

        if candidate_pool is not None:
            pool_set = set(str(x) for x in candidate_pool)
            # Find past items that are in the candidate pool
            user_recs = [str(x) for x in past_items if str(x) in pool_set]

            if len(user_recs) < k:
                # Fill remainder from candidate pool using global popularity
                remaining = [x for x in candidate_pool if str(x) not in set(user_recs)]
                fallback = self.global_popularity.recommend(user_id, k=k - len(user_recs), candidate_pool=remaining)
                user_recs.extend(fallback)
            return user_recs[:k]

        user_recs = list(past_items)
        if len(user_recs) < k:
            fallback = self.global_popularity.recommend(user_id, k=k - len(user_recs))
            for item in fallback:
                if item not in user_recs:
                    user_recs.append(item)
                if len(user_recs) == k:
                    break
        return user_recs[:k]


class RandomRecommender(BaseRecommender):
    """
    Uniform Random Recommender.
    Selects items randomly without replacement from the candidate pool or catalog.
    Provides the empirical and theoretical lower-bound baseline anchor.
    """

    def __init__(
        self,
        item_col: str = "article_id",
        seed: int = 42,
    ):
        self.item_col = item_col
        self.seed = seed
        self.all_items: np.ndarray = np.array([])
        self.rng = np.random.default_rng(seed)

    def fit(self, train_df: pd.DataFrame) -> "RandomRecommender":
        clean_items = train_df[self.item_col].dropna().astype(str).str.strip().unique()
        self.all_items = np.array(clean_items)
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_pool: Optional[List[str]] = None,
    ) -> List[str]:
        items_to_sample = candidate_pool if candidate_pool is not None else self.all_items
        if len(items_to_sample) == 0:
            return []

        sample_k = min(k, len(items_to_sample))
        chosen = self.rng.choice(items_to_sample, size=sample_k, replace=False)
        return [str(x) for x in chosen]


class BenchmarkHarness:
    """
    Comparative Recommendation Evaluation Benchmark Suite.
    Evaluates multiple recommendation models/baselines against the same test dataset,
    aggregates ranking metrics (Recall@K, NDCG@K, MRR@K, HitRate@K), and builds
    a comparative leaderboard.
    """

    def __init__(
        self,
        ground_truth: Dict[str, List[str]],
        candidate_pools: Optional[Dict[str, List[str]]] = None,
        k_values: Sequence[int] = (5, 10, 20),
    ):
        """
        Args:
            ground_truth: Dict mapping user_id to actual positive item IDs.
            candidate_pools: Optional candidate pool dictionary per user.
            k_values: Sequence of K thresholds to evaluate.
        """
        self.ground_truth = ground_truth
        self.candidate_pools = candidate_pools
        self.k_values = sorted(list(k_values))
        self.evaluator = RetrievalEvaluator(k_values=self.k_values)
        self.models: Dict[str, BaseRecommender] = {}
        self.precomputed_predictions: Dict[str, Dict[str, List[str]]] = {}

    def register_model(self, model_name: str, model: BaseRecommender) -> "BenchmarkHarness":
        """Registers a BaseRecommender model instance."""
        self.models[model_name] = model
        return self

    def register_predictions(
        self,
        model_name: str,
        predictions: Dict[str, List[str]],
    ) -> "BenchmarkHarness":
        """Registers pre-generated recommendations (e.g. from Two-Tower model)."""
        self.precomputed_predictions[model_name] = predictions
        return self

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs evaluation across all registered models and predictions.

        Returns:
            pd.DataFrame: Leaderboard table with columns:
                ['Model', 'Recall@5', 'NDCG@5', 'Recall@10', 'NDCG@10', 'MRR@10', 'HitRate@10', ...]
        """
        records = []
        user_ids = list(self.ground_truth.keys())
        max_k = max(self.k_values) if self.k_values else 10

        # 1. Evaluate registered recommender models
        for name, model in self.models.items():
            logger.info(f"Evaluating benchmark model: {name}...")
            preds = model.predict_batch(
                user_ids=user_ids,
                k=max_k,
                candidate_pools=self.candidate_pools,
            )
            metrics = self.evaluator.evaluate_predictions(self.ground_truth, preds)
            row = {"Model": name}
            row.update(metrics)
            records.append(row)

        # 2. Evaluate precomputed model predictions
        for name, preds in self.precomputed_predictions.items():
            logger.info(f"Evaluating precomputed model predictions: {name}...")
            metrics = self.evaluator.evaluate_predictions(self.ground_truth, preds)
            row = {"Model": name}
            row.update(metrics)
            records.append(row)

        leaderboard = pd.DataFrame(records)

        # Sort leaderboard by NDCG at main K (e.g. 10) or Recall at main K
        sort_metric = f"ndcg@{min(10, max_k)}"
        if sort_metric in leaderboard.columns:
            leaderboard = leaderboard.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)

        return leaderboard

    def print_leaderboard(self, leaderboard: pd.DataFrame) -> None:
        """Prints formatted leaderboard table to stdout."""
        print("\n" + "=" * 80)
        print("RECOMMENDATION BENCHMARK LEADERBOARD")
        print("=" * 80)
        try:
            print(leaderboard.to_markdown(index=False))
        except (ImportError, ModuleNotFoundError):
            print(leaderboard.to_string(index=False))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    print("=== RUNNING DAY 3 BASELINES & BENCHMARK DEMO ===")
    
    # 1. Mock Training interactions
    mock_train = pd.DataFrame({
        "customer_id": ["u1", "u1", "u2", "u2", "u3", "u1", "u4", "u2"],
        "article_id": ["item_popular", "item_recent", "item_popular", "item_mid", "item_popular", "item_old", "item_mid", "item_popular"],
        "t_dat": [
            "2026-08-01", "2026-08-15", "2026-08-02", "2026-08-10",
            "2026-08-03", "2026-07-01", "2026-08-05", "2026-08-14"
        ]
    })

    # 2. Mock Test ground truth and candidate pool
    mock_test_gt = {
        "u1": ["item_recent"],
        "u2": ["item_popular"],
        "u3": ["item_mid"],
        "u4": ["item_popular"],
    }
    candidate_pools = {
        "u1": ["item_recent", "item_popular", "item_old", "item_mid", "item_random1"],
        "u2": ["item_popular", "item_mid", "item_old", "item_random2", "item_random3"],
        "u3": ["item_mid", "item_popular", "item_recent", "item_random4", "item_random5"],
        "u4": ["item_popular", "item_recent", "item_mid", "item_random6", "item_random7"],
    }

    # 3. Instantiate and fit baselines
    pop_model = PopularityRecommender().fit(mock_train)
    rec_model = RecentPopularityRecommender(half_life_days=7.0).fit(mock_train)
    hist_model = UserHistoryRecommender().fit(mock_train)
    rand_model = RandomRecommender(seed=42).fit(mock_train)

    # 4. Run Benchmark Harness
    harness = BenchmarkHarness(
        ground_truth=mock_test_gt,
        candidate_pools=candidate_pools,
        k_values=[1, 3, 5],
    )
    harness.register_model("Random Baseline", rand_model)
    harness.register_model("Global Popularity", pop_model)
    harness.register_model("Recent Trend Popularity", rec_model)
    harness.register_model("User History / Repeat", hist_model)

    # Simulate a mock Two-Tower Neural Model prediction (high precision)
    simulated_two_tower = {
        "u1": ["item_recent", "item_popular"],
        "u2": ["item_popular", "item_mid"],
        "u3": ["item_mid", "item_popular"],
        "u4": ["item_popular", "item_mid"],
    }
    harness.register_predictions("Two-Tower Neural (Simulated)", simulated_two_tower)

    leaderboard = harness.run_benchmark()
    harness.print_leaderboard(leaderboard)

    print("DAY 3 BASELINE IMPLEMENTATION & BENCHMARK EXECUTION COMPLETED!")
