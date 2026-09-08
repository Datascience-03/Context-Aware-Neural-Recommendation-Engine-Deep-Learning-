"""
Day 2: Evaluation Dataset Generation & Top-K Retrieval Evaluator Pipeline.
Provides EvaluationDataset for preparing test evaluation pairs and candidate pools,
and RetrievalEvaluator for evaluating recommendation models and embedding representations.
"""

import os
import sys
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

logger = logging.getLogger(__name__)


class EvaluationDataset:
    """
    Manages evaluation dataset preparation, splitting, and candidate pool creation
    for context-aware recommendation evaluation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_col: str = "customer_id",
        item_col: str = "article_id",
        time_col: Optional[str] = "t_dat",
    ):
        """
        Initialize evaluation dataset from interactions dataframe.

        Args:
            df: Interaction records (user, item, timestamp, context).
            user_col: Column name identifying user/query.
            item_col: Column name identifying candidate item/article.
            time_col: Optional timestamp or date column for temporal sorting.
        """
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        # Clean null values in ID columns
        clean_df = df.dropna(subset=[user_col, item_col]).copy()
        clean_df[user_col] = clean_df[user_col].astype(str).str.strip()
        clean_df[item_col] = clean_df[item_col].astype(str).str.strip()
        self.df = clean_df

        # Unique catalog items across the dataset
        self.all_items: np.ndarray = np.array(sorted(self.df[item_col].unique()))

        # Build ground-truth dictionary: user -> list of interacted positive items
        self.ground_truth: Dict[str, List[str]] = (
            self.df.groupby(user_col)[item_col].apply(lambda s: list(dict.fromkeys(s))).to_dict()
        )

    @classmethod
    def create_temporal_split(
        cls,
        df: pd.DataFrame,
        time_col: str = "t_dat",
        test_ratio: float = 0.2,
        user_col: str = "customer_id",
        item_col: str = "article_id",
    ) -> Tuple["EvaluationDataset", "EvaluationDataset"]:
        """
        Splits interactions chronologically into Train and Test datasets.
        Crucial for recommendation to avoid temporal data leakage.

        Args:
            df: Interaction dataframe.
            time_col: Timestamp or date column.
            test_ratio: Fraction of interactions allocated to test split.
            user_col: User identifier column.
            item_col: Item identifier column.

        Returns:
            Tuple[EvaluationDataset, EvaluationDataset]: (train_dataset, test_dataset)
        """
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found in dataframe.")

        sorted_df = df.sort_values(by=time_col).reset_index(drop=True)
        split_idx = int(len(sorted_df) * (1.0 - test_ratio))

        train_df = sorted_df.iloc[:split_idx]
        test_df = sorted_df.iloc[split_idx:]

        return (
            cls(train_df, user_col=user_col, item_col=item_col, time_col=time_col),
            cls(test_df, user_col=user_col, item_col=item_col, time_col=time_col),
        )

    @classmethod
    def create_leave_k_out_split(
        cls,
        df: pd.DataFrame,
        k: int = 1,
        user_col: str = "customer_id",
        item_col: str = "article_id",
        time_col: str = "t_dat",
    ) -> Tuple["EvaluationDataset", "EvaluationDataset"]:
        """
        Leave-last-K-out splitting per user.
        For each user, the latest K interactions form the test set; earlier interactions form train set.

        Args:
            df: Interaction dataframe.
            k: Number of most recent interactions per user for testing.
            user_col: User column name.
            item_col: Item column name.
            time_col: Timestamp column name.

        Returns:
            Tuple[EvaluationDataset, EvaluationDataset]: (train_dataset, test_dataset)
        """
        sorted_df = df.sort_values(by=[user_col, time_col]).reset_index(drop=True)

        # Assign rank within each user from latest (rank 1) to oldest
        sorted_df["_rank_desc"] = (
            sorted_df.groupby(user_col).cumcount(ascending=False) + 1
        )

        test_df = sorted_df[sorted_df["_rank_desc"] <= k].drop(columns=["_rank_desc"])
        train_df = sorted_df[sorted_df["_rank_desc"] > k].drop(columns=["_rank_desc"])

        return (
            cls(train_df, user_col=user_col, item_col=item_col, time_col=time_col),
            cls(test_df, user_col=user_col, item_col=item_col, time_col=time_col),
        )

    def generate_candidate_pools(
        self,
        num_negatives: int = 100,
        seed: int = 42,
        user_history: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generates fixed candidate pools for each evaluation user:
        candidate_pool = ground_truth_positives + num_negatives randomly sampled unobserved items.

        Standard benchmark methodology (e.g. 1 positive + 99 negatives).

        Args:
            num_negatives: Number of negative/unobserved candidate items to sample per user.
            seed: Random seed for reproducibility.
            user_history: Optional dictionary of prior training interactions to exclude.

        Returns:
            Dict[str, List[str]]: Mapping from user_id to list of candidate item IDs.
        """
        rng = np.random.default_rng(seed)
        all_items_set = set(self.all_items)
        candidate_pools: Dict[str, List[str]] = {}

        for user_id, pos_items in self.ground_truth.items():
            excluded_items = set(pos_items)
            if user_history and user_id in user_history:
                excluded_items.update(user_history[user_id])

            available_negatives = np.array(list(all_items_set - excluded_items))

            if len(available_negatives) == 0:
                # Fallback if catalog is tiny
                candidate_pools[user_id] = list(pos_items)
                continue

            sample_size = min(num_negatives, len(available_negatives))
            sampled_negatives = [str(x) for x in rng.choice(available_negatives, size=sample_size, replace=False)]

            # Combine true positives with sampled negatives and shuffle
            pool = [str(x) for x in pos_items] + sampled_negatives
            rng.shuffle(pool)
            candidate_pools[user_id] = pool

        return candidate_pools

    def get_test_users(self) -> List[str]:
        """Returns list of all test user IDs."""
        return list(self.ground_truth.keys())

    def get_ground_truth(self, user_id: str) -> List[str]:
        """Returns list of positive items for given user."""
        return self.ground_truth.get(str(user_id), [])


class RetrievalEvaluator:
    """
    Candidate Retrieval Evaluator for Two-Tower / Neural Recommendation Models.
    Computes Recall@K, NDCG@K, MRR@K, and Hit Rate@K across top-K recommendations.
    """

    def __init__(self, k_values: Sequence[int] = (5, 10, 20)):
        """
        Args:
            k_values: Sequence of K thresholds to evaluate (e.g. 5, 10, 20).
        """
        self.k_values = sorted(list(k_values))
        self.max_k = max(self.k_values) if self.k_values else 10

    def evaluate_predictions(
        self,
        ground_truth: Dict[str, List[str]],
        predictions: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Evaluates pre-ranked recommendation lists against ground truth.

        Args:
            ground_truth: Dict mapping user_id to actual positive items.
            predictions: Dict mapping user_id to ordered list of recommended items.

        Returns:
            Dict[str, float]: Aggregated metrics across all evaluated users.
        """
        common_users = [u for u in ground_truth if u in predictions]

        if not common_users:
            logger.warning("No overlapping users found between ground truth and predictions.")
            return {f"{m}@{k}": 0.0 for k in self.k_values for m in ["recall", "ndcg", "mrr", "hit_rate"]}

        actual_list = [ground_truth[u] for u in common_users]
        predicted_list = [predictions[u] for u in common_users]

        return evaluate_batch_metrics(actual_list, predicted_list, k_list=self.k_values)

    def evaluate_embeddings(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        user_ids: Sequence[str],
        item_ids: Sequence[str],
        ground_truth: Dict[str, List[str]],
        candidate_pools: Optional[Dict[str, List[str]]] = None,
        filter_train_history: Optional[Dict[str, Set[str]]] = None,
        batch_size: int = 512,
    ) -> Dict[str, float]:
        """
        Evaluates Two-Tower retrieval directly from Query Tower and Candidate Tower embeddings.
        Computes cosine similarity / dot-product scores: S = U * V^T,
        retrieves top-K items, and calculates retrieval ranking metrics.

        Args:
            user_embeddings: Matrix of shape (N_users, embedding_dim).
            item_embeddings: Matrix of shape (N_items, embedding_dim).
            user_ids: List of user IDs corresponding to user_embeddings rows.
            item_ids: List of item IDs corresponding to item_embeddings rows.
            ground_truth: Dict mapping user_id to ground truth positive item IDs.
            candidate_pools: Optional dictionary restricting candidates per user.
            filter_train_history: Optional dictionary of historical items to mask out.
            batch_size: Number of users to score simultaneously.

        Returns:
            Dict[str, float]: Evaluation metrics across all test users.
        """
        user_ids_str = [str(u) for u in user_ids]
        item_ids_str = [str(i) for i in item_ids]
        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_ids_str)}

        predictions: Dict[str, List[str]] = {}

        # Normalize embeddings for cosine similarity retrieval
        user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
        user_norms = np.where(user_norms == 0, 1e-12, user_norms)
        norm_user_emb = user_embeddings / user_norms

        item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        item_norms = np.where(item_norms == 0, 1e-12, item_norms)
        norm_item_emb = item_embeddings / item_norms

        num_users = len(user_ids_str)

        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)
            batch_u_ids = user_ids_str[start_idx:end_idx]
            batch_u_emb = norm_user_emb[start_idx:end_idx]

            # Batch score matrix: shape (batch_len, N_items)
            scores = np.matmul(batch_u_emb, norm_item_emb.T)

            for b_i, u_id in enumerate(batch_u_ids):
                user_scores = scores[b_i].copy()

                # If candidate pool is specified for this user
                if candidate_pools and u_id in candidate_pools:
                    allowed_pool = candidate_pools[u_id]
                    allowed_indices = [item_id_to_idx[item] for item in allowed_pool if item in item_id_to_idx]

                    if not allowed_indices:
                        predictions[u_id] = []
                        continue

                    pool_scores = user_scores[allowed_indices]
                    top_k_count = min(self.max_k, len(allowed_indices))
                    top_pool_subindices = np.argpartition(-pool_scores, top_k_count - 1)[:top_k_count]
                    # Sort top-K subset exactly
                    top_sorted = top_pool_subindices[np.argsort(-pool_scores[top_pool_subindices])]
                    predictions[u_id] = [item_ids_str[allowed_indices[idx]] for idx in top_sorted]

                else:
                    # Full catalog retrieval
                    if filter_train_history and u_id in filter_train_history:
                        history_indices = [
                            item_id_to_idx[item]
                            for item in filter_train_history[u_id]
                            if item in item_id_to_idx
                        ]
                        user_scores[history_indices] = -np.inf

                    top_k_count = min(self.max_k, len(user_scores))
                    top_indices = np.argpartition(-user_scores, top_k_count - 1)[:top_k_count]
                    top_sorted = top_indices[np.argsort(-user_scores[top_indices])]
                    predictions[u_id] = [item_ids_str[idx] for idx in top_sorted]

        return self.evaluate_predictions(ground_truth, predictions)

    def evaluate_scorer(
        self,
        scoring_fn: Callable[[str, List[str]], Sequence[float]],
        ground_truth: Dict[str, List[str]],
        candidate_pools: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """
        Evaluates an arbitrary scoring function (e.g. Model inference callback).

        Args:
            scoring_fn: Callable(user_id, candidate_item_ids) -> list of numeric scores.
            ground_truth: Ground truth actual positive items.
            candidate_pools: Candidate item pools to rank per user.

        Returns:
            Dict[str, float]: Metric results.
        """
        predictions: Dict[str, List[str]] = {}

        for user_id, candidates in candidate_pools.items():
            if not candidates:
                predictions[user_id] = []
                continue

            scores = np.array(scoring_fn(user_id, candidates))
            top_k_count = min(self.max_k, len(candidates))
            top_subindices = np.argpartition(-scores, top_k_count - 1)[:top_k_count]
            top_sorted = top_subindices[np.argsort(-scores[top_subindices])]
            predictions[user_id] = [candidates[idx] for idx in top_sorted]

        return self.evaluate_predictions(ground_truth, predictions)

    def generate_evaluation_report_df(
        self,
        ground_truth: Dict[str, List[str]],
        predictions: Dict[str, List[str]],
        k_eval: int = 10,
    ) -> pd.DataFrame:
        """
        Generates a per-user diagnostic report dataframe for error analysis.

        Args:
            ground_truth: Actual positive items.
            predictions: Predicted item recommendations.
            k_eval: Specific K threshold to record in report.

        Returns:
            pd.DataFrame: Table with user_id, num_ground_truth, top_k_recs, recall@k, ndcg@k, hit@k.
        """
        rows = []
        for u_id, actual in ground_truth.items():
            preds = predictions.get(u_id, [])
            rec = recall_at_k(actual, preds, k=k_eval)
            ndcg = ndcg_at_k(actual, preds, k=k_eval)
            mrr = mrr_at_k(actual, preds, k=k_eval)
            hit = hit_rate_at_k(actual, preds, k=k_eval)

            rows.append({
                "user_id": u_id,
                "num_ground_truth": len(actual),
                f"top_{k_eval}_predictions": preds[:k_eval],
                f"recall@{k_eval}": round(rec, 4),
                f"ndcg@{k_eval}": round(ndcg, 4),
                f"mrr@{k_eval}": round(mrr, 4),
                f"hit@{k_eval}": int(hit),
            })

        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Sanity demonstration of Day 2 Evaluator Pipeline
    print("=== RUNNING DAY 2 EVALUATOR SANITY DEMO ===")
    
    # 1. Create mock interaction dataframe
    mock_data = {
        "customer_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
        "article_id": ["a10", "a20", "a20", "a30", "a10", "a40", "a50", "a60"],
        "t_dat": [
            "2026-08-01", "2026-08-02", "2026-08-01", "2026-08-03",
            "2026-08-01", "2026-08-04", "2026-08-02", "2026-08-05"
        ]
    }
    df_interactions = pd.DataFrame(mock_data)

    # 2. Build EvaluationDataset
    eval_dataset = EvaluationDataset(df_interactions)
    print(f"Total catalog items: {len(eval_dataset.all_items)}")
    print(f"Test users count: {len(eval_dataset.get_test_users())}")

    # 3. Generate candidate pools (1 positive + 4 negatives)
    candidate_pools = eval_dataset.generate_candidate_pools(num_negatives=4, seed=42)
    print("Sample candidate pool for user u1:", candidate_pools["u1"])

    # 4. Initialize Evaluator
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

    # 5. Evaluate dummy predictions
    mock_predictions = {
        "u1": ["a10", "a20", "a99"],
        "u2": ["a99", "a20", "a30"],
        "u3": ["a99", "a98", "a97"],
        "u4": ["a50", "a60", "a10"],
    }
    results = evaluator.evaluate_predictions(eval_dataset.ground_truth, mock_predictions)
    print("\n--- Evaluation Benchmark Results ---")
    for metric, val in results.items():
        print(f"  {metric:<15}: {val:.4f}")

    # 6. Generate detailed diagnostic report
    report_df = evaluator.generate_evaluation_report_df(
        eval_dataset.ground_truth, mock_predictions, k_eval=3
    )
    print("\n--- Diagnostic User Report (Top 3) ---")
    print(report_df.head())
    print("\nDAY 2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
