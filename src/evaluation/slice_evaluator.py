"""
Day 5: Context-Slice & Recommendation Diversity/Coverage Evaluation.
Provides beyond-accuracy metrics (Catalog Coverage, Intra-List Diversity, Prediction Entropy,
Gini Coefficient, Novelty) and SliceEvaluator for slice-based fairness, consistency,
and performance disparity analysis across contextual segments (season, weekend, channel, etc.).
"""

import os
import sys
import math
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
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


# =====================================================================
# 1. Beyond-Accuracy Recommendation Metrics
# =====================================================================

def catalog_coverage(
    predicted_lists: Sequence[Sequence[Any]],
    catalog_items: Sequence[Any],
    k: int = 10,
) -> float:
    """
    Compute Catalog Coverage @ K: fraction of catalog items recommended at least once.

    CatalogCoverage@K = |U_{u} (predicted_u[:k] ∩ catalog)| / |catalog|

    Args:
        predicted_lists: List of recommended item ID sequences for each query session.
        catalog_items: Sequence of all unique catalog item IDs.
        k: Top-K rank cutoff.

    Returns:
        float: Coverage proportion in [0.0, 1.0].
    """
    if not catalog_items or not predicted_lists or k <= 0:
        return 0.0

    catalog_set = set(catalog_items)
    if not catalog_set:
        return 0.0

    recommended_unique: Set[Any] = set()
    for preds in predicted_lists:
        for item in preds[:k]:
            if item in catalog_set:
                recommended_unique.add(item)

    return float(len(recommended_unique) / len(catalog_set))


def intra_list_diversity(
    predicted_lists: Sequence[Sequence[Any]],
    item_embeddings: Optional[Dict[Any, np.ndarray]] = None,
    item_categories: Optional[Dict[Any, Any]] = None,
    k: int = 10,
) -> float:
    """
    Compute average Intra-List Diversity (ILD) @ K across query sessions.
    Measures the average pairwise dissimilarity among items in a user's top-K recommendation list.

    If item_embeddings is provided, dissimilarity is cosine distance: 1 - cosine_similarity(e_i, e_j).
    If item_categories is provided, dissimilarity is categorical mismatch: 1.0 if cat_i != cat_j else 0.0.

    Args:
        predicted_lists: List of recommended item ID sequences.
        item_embeddings: Optional dictionary mapping item_id -> 1D embedding vector.
        item_categories: Optional dictionary mapping item_id -> category/genre label.
        k: Top-K rank cutoff.

    Returns:
        float: Average intra-list diversity score in [0.0, 1.0].
    """
    if not predicted_lists or k <= 1:
        return 0.0

    session_ilds: List[float] = []

    for preds in predicted_lists:
        top_k = list(preds[:k])
        n_items = len(top_k)
        if n_items <= 1:
            continue

        pairwise_distances = []

        if item_embeddings is not None:
            # Pairwise cosine distance
            for i in range(n_items):
                item_i = top_k[i]
                if item_i not in item_embeddings:
                    continue
                emb_i = item_embeddings[item_i]
                norm_i = np.linalg.norm(emb_i)
                if norm_i == 0:
                    continue

                for j in range(i + 1, n_items):
                    item_j = top_k[j]
                    if item_j not in item_embeddings:
                        continue
                    emb_j = item_embeddings[item_j]
                    norm_j = np.linalg.norm(emb_j)
                    if norm_j == 0:
                        continue

                    cos_sim = float(np.dot(emb_i, emb_j) / (norm_i * norm_j))
                    # Clip between -1 and 1 for numerical safety
                    cos_sim = max(-1.0, min(1.0, cos_sim))
                    # Normalized cosine distance in [0, 1]
                    cos_dist = (1.0 - cos_sim) / 2.0
                    pairwise_distances.append(cos_dist)

        elif item_categories is not None:
            # Pairwise category mismatch distance
            for i in range(n_items):
                cat_i = item_categories.get(top_k[i])
                for j in range(i + 1, n_items):
                    cat_j = item_categories.get(top_k[j])
                    if cat_i is not None and cat_j is not None:
                        pairwise_distances.append(1.0 if cat_i != cat_j else 0.0)

        else:
            # Default ID-based mismatch: distinct items are 1.0 distance
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    pairwise_distances.append(1.0 if top_k[i] != top_k[j] else 0.0)

        if pairwise_distances:
            session_ilds.append(float(np.mean(pairwise_distances)))

    return float(np.mean(session_ilds)) if session_ilds else 0.0


def prediction_entropy(
    predicted_lists: Sequence[Sequence[Any]],
    k: int = 10,
) -> float:
    """
    Compute Shannon Entropy of item recommendation distribution across all users/sessions.
    Higher entropy indicates a more balanced recommendation distribution (less popularity bias).

    H = - sum_{i} p(i) * log2(p(i))

    Args:
        predicted_lists: List of recommended item ID sequences.
        k: Top-K rank cutoff.

    Returns:
        float: Shannon entropy in bits (>= 0.0).
    """
    if not predicted_lists or k <= 0:
        return 0.0

    counts: Counter = Counter()
    total_recs = 0
    for preds in predicted_lists:
        for item in preds[:k]:
            counts[item] += 1
            total_recs += 1

    if total_recs == 0:
        return 0.0

    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total_recs
        if p > 0:
            entropy -= p * math.log2(p)

    return float(entropy)


def gini_coefficient(
    predicted_lists: Sequence[Sequence[Any]],
    catalog_items: Optional[Sequence[Any]] = None,
    k: int = 10,
) -> float:
    """
    Compute the Gini Coefficient of recommendation distribution across items.
    Measures inequality of recommendation opportunities across catalog items.
    0.0 represents perfect equality; 1.0 represents maximal concentration (single item).

    Args:
        predicted_lists: List of recommended item ID sequences.
        catalog_items: Optional sequence of all catalog items. If provided, unrecommended
                       catalog items are counted with frequency 0.
        k: Top-K cutoff.

    Returns:
        float: Gini coefficient in [0.0, 1.0].
    """
    if not predicted_lists or k <= 0:
        return 0.0

    counts: Counter = Counter()
    for preds in predicted_lists:
        for item in preds[:k]:
            counts[item] += 1

    if catalog_items is not None:
        catalog_set = set(catalog_items)
        # Ensure all catalog items are represented
        frequencies = np.array([counts.get(item, 0) for item in catalog_set], dtype=np.float64)
    else:
        frequencies = np.array(list(counts.values()), dtype=np.float64)

    n = len(frequencies)
    if n <= 1:
        return 0.0

    total = np.sum(frequencies)
    if total == 0:
        return 0.0

    # Sort frequencies in ascending order
    sorted_freqs = np.sort(frequencies)
    index = np.arange(1, n + 1)
    # Standard Gini formula: (2 * sum(i * y_i) - (n + 1) * sum(y_i)) / (n * sum(y_i))
    gini = (2.0 * np.sum(index * sorted_freqs) - (n + 1) * total) / (n * total)
    return float(max(0.0, min(1.0, gini)))


def novelty_at_k(
    predicted_lists: Sequence[Sequence[Any]],
    item_popularities: Dict[Any, float],
    k: int = 10,
    eps: float = 1e-12,
) -> float:
    """
    Compute average Novelty @ K (Self-Information) across recommendations.
    Novelty measures how unexpected / non-popular the recommended items are:

    Novelty@K = (1 / |U|) * sum_{u} (1 / K) * sum_{i in recs[:K]} -log2(P(i))

    where P(i) is the historical interaction frequency of item i.

    Args:
        predicted_lists: List of recommended item ID sequences.
        item_popularities: Dictionary mapping item_id -> historical popularity probability P(i) in (0, 1].
        k: Top-K rank cutoff.
        eps: Small floor value for unobserved/new items.

    Returns:
        float: Average self-information in bits (>= 0.0).
    """
    if not predicted_lists or not item_popularities or k <= 0:
        return 0.0

    session_novelties: List[float] = []

    for preds in predicted_lists:
        top_k = preds[:k]
        if not top_k:
            continue

        item_novelties = []
        for item in top_k:
            prob = item_popularities.get(item, eps)
            prob = max(prob, eps)
            item_novelties.append(-math.log2(prob))

        session_novelties.append(float(np.mean(item_novelties)))

    return float(np.mean(session_novelties)) if session_novelties else 0.0


# =====================================================================
# 2. Context-Slice Evaluator
# =====================================================================

class SliceEvaluator:
    """
    Contextual Slice & Fairness Evaluator for Two-Tower Recommendation Systems.
    Evaluates ranking performance and beyond-accuracy metrics partitioned across
    context dimensions (e.g. season, weekend, sales channel, customer age group).
    Detects severe slice degradation and performance disparity across subgroups.
    """

    def __init__(
        self,
        catalog_items: Optional[Sequence[Any]] = None,
        item_embeddings: Optional[Dict[Any, np.ndarray]] = None,
        item_popularities: Optional[Dict[Any, float]] = None,
    ):
        """
        Initialize SliceEvaluator.

        Args:
            catalog_items: Complete sequence of valid catalog item identifiers.
            item_embeddings: Mapping from item_id to embedding vector for ILD calculation.
            item_popularities: Mapping from item_id to interaction probability for Novelty calculation.
        """
        self.catalog_items = list(catalog_items) if catalog_items is not None else []
        self.item_embeddings = item_embeddings
        self.item_popularities = item_popularities

        self.sessions: List[Dict[str, Any]] = []

    def clear(self) -> None:
        """Reset registered evaluation sessions."""
        self.sessions = []

    def add_session(
        self,
        session_id: Any,
        actual: Sequence[Any],
        predicted: Sequence[Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an individual evaluation session with its ground truth, predictions, and context.

        Args:
            session_id: Unique query/session or customer identifier.
            actual: Sequence of relevant ground-truth item IDs.
            predicted: Ordered sequence of predicted item recommendations.
            context: Dictionary of contextual features (e.g. {'season': 'Winter', 'is_weekend': 1}).
        """
        self.sessions.append({
            "session_id": session_id,
            "actual": list(actual),
            "predicted": list(predicted),
            "context": dict(context) if context else {},
        })

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        actual_col: str,
        predicted_col: str,
        context_cols: Sequence[str],
        session_id_col: Optional[str] = None,
    ) -> "SliceEvaluator":
        """
        Load evaluation sessions directly from a Pandas DataFrame.

        Args:
            df: DataFrame containing predictions, ground truth, and context columns.
            actual_col: Column with actual relevant items (list or single item).
            predicted_col: Column with predicted items (ordered list).
            context_cols: List of column names to store as context attributes.
            session_id_col: Optional column name for session/customer ID.

        Returns:
            self
        """
        for idx, row in df.iterrows():
            sess_id = row[session_id_col] if session_id_col and session_id_col in df.columns else idx
            raw_actual = row[actual_col]
            actual = raw_actual if isinstance(raw_actual, (list, tuple, set, np.ndarray)) else [raw_actual]
            predicted = row[predicted_col]
            if not isinstance(predicted, (list, tuple, np.ndarray)):
                predicted = [predicted]

            ctx = {col: row[col] for col in context_cols if col in df.columns}
            self.add_session(session_id=sess_id, actual=actual, predicted=predicted, context=ctx)

        return self

    @property
    def num_sessions(self) -> int:
        """Total registered sessions."""
        return len(self.sessions)

    def evaluate_slice(
        self,
        slice_feature: str,
        k_values: Sequence[int] = (5, 10, 20),
        min_slice_size: int = 1,
    ) -> pd.DataFrame:
        """
        Evaluate ranking and beyond-accuracy metrics partitioned by a contextual feature.

        Args:
            slice_feature: Name of contextual attribute in context dictionary (e.g. 'season', 'is_weekend').
            k_values: List of K cutoffs to evaluate.
            min_slice_size: Minimum number of sessions required to include a slice.

        Returns:
            pd.DataFrame summarizing metrics per slice.
        """
        if not self.sessions:
            return pd.DataFrame()

        # Group sessions by slice value
        grouped_sessions: Dict[Any, List[Dict[str, Any]]] = {}
        for s in self.sessions:
            val = s["context"].get(slice_feature, "Unknown")
            if pd.isna(val):
                val = "Missing"
            grouped_sessions.setdefault(val, []).append(s)

        primary_k = k_values[0] if k_values else 10
        rows = []

        total_sessions = len(self.sessions)

        for slice_val, sess_list in grouped_sessions.items():
            slice_size = len(sess_list)
            if slice_size < min_slice_size:
                continue

            slice_share = slice_size / total_sessions
            actual_list = [s["actual"] for s in sess_list]
            pred_list = [s["predicted"] for s in sess_list]

            # Ranking accuracy metrics
            batch_metrics = evaluate_batch_metrics(actual_list, pred_list, k_list=list(k_values))

            # Beyond-accuracy metrics
            cov = catalog_coverage(pred_list, self.catalog_items, k=primary_k) if self.catalog_items else 0.0
            ild = intra_list_diversity(pred_list, item_embeddings=self.item_embeddings, k=primary_k)
            ent = prediction_entropy(pred_list, k=primary_k)
            gini = gini_coefficient(pred_list, catalog_items=self.catalog_items, k=primary_k)

            row: Dict[str, Any] = {
                "slice_feature": slice_feature,
                "slice_value": slice_val,
                "session_count": slice_size,
                "slice_share": slice_share,
            }

            # Add ranking metrics
            for k in k_values:
                row[f"recall@{k}"] = batch_metrics[f"recall@{k}"]
                row[f"ndcg@{k}"] = batch_metrics[f"ndcg@{k}"]
                row[f"mrr@{k}"] = batch_metrics[f"mrr@{k}"]
                row[f"hit_rate@{k}"] = batch_metrics[f"hit_rate@{k}"]

            # Add beyond-accuracy metrics
            row[f"coverage@{primary_k}"] = cov
            row[f"ild@{primary_k}"] = ild
            row[f"entropy@{primary_k}"] = ent
            row[f"gini@{primary_k}"] = gini

            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="session_count", ascending=False).reset_index(drop=True)
        return df

    def evaluate_all_slices(
        self,
        slice_features: Sequence[str],
        k_values: Sequence[int] = (5, 10, 20),
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate multiple context features and return dictionary of slice DataFrames.

        Args:
            slice_features: List of context feature names.
            k_values: Top-K cutoffs.

        Returns:
            Dict[slice_feature, pd.DataFrame]
        """
        results = {}
        for feat in slice_features:
            df = self.evaluate_slice(feat, k_values=k_values)
            results[feat] = df
        return results

    def compute_slice_disparity(
        self,
        slice_df: pd.DataFrame,
        metric: str = "ndcg@10",
        disparity_threshold: float = 0.60,
    ) -> Dict[str, Any]:
        """
        Compute fairness disparity across slices for a specific evaluation metric.

        Disparity Ratio = min(metric) / max(metric)
        A disparity ratio < 0.60 highlights substantial performance degradation
        on underserved contextual segments.

        Args:
            slice_df: DataFrame returned by evaluate_slice.
            metric: Target metric column name (e.g. 'ndcg@10', 'recall@10').
            disparity_threshold: Threshold below which disparity is flagged as a concern.

        Returns:
            Dictionary with disparity statistics.
        """
        if slice_df.empty or metric not in slice_df.columns:
            return {
                "metric": metric,
                "min_slice": None,
                "min_value": 0.0,
                "max_slice": None,
                "max_value": 0.0,
                "disparity_ratio": 1.0,
                "disparity_flagged": False,
                "std_across_slices": 0.0,
            }

        valid_df = slice_df.dropna(subset=[metric])
        if valid_df.empty:
            return {"metric": metric, "disparity_ratio": 1.0, "disparity_flagged": False}

        min_idx = valid_df[metric].idxmin()
        max_idx = valid_df[metric].idxmax()

        min_val = float(valid_df.loc[min_idx, metric])
        max_val = float(valid_df.loc[max_idx, metric])

        min_slice = valid_df.loc[min_idx, "slice_value"]
        max_slice = valid_df.loc[max_idx, "slice_value"]

        ratio = (min_val / max_val) if max_val > 0 else 1.0
        flagged = (ratio < disparity_threshold) and (max_val > 0)
        std_val = float(valid_df[metric].std()) if len(valid_df) > 1 else 0.0

        return {
            "metric": metric,
            "min_slice": min_slice,
            "min_value": min_val,
            "max_slice": max_slice,
            "max_value": max_val,
            "disparity_ratio": ratio,
            "disparity_flagged": flagged,
            "std_across_slices": std_val,
        }

    def evaluate_global_beyond_accuracy(
        self,
        k_values: Sequence[int] = (5, 10, 20),
    ) -> Dict[str, float]:
        """
        Calculate global catalog-wide beyond-accuracy metrics.

        Args:
            k_values: List of K cutoffs to evaluate.

        Returns:
            Dictionary of diversity, coverage, entropy, and novelty metrics.
        """
        if not self.sessions:
            return {}

        all_preds = [s["predicted"] for s in self.sessions]
        results: Dict[str, float] = {}

        for k in k_values:
            if self.catalog_items:
                results[f"catalog_coverage@{k}"] = catalog_coverage(all_preds, self.catalog_items, k=k)
                results[f"gini_coefficient@{k}"] = gini_coefficient(all_preds, catalog_items=self.catalog_items, k=k)
            else:
                results[f"catalog_coverage@{k}"] = 0.0
                results[f"gini_coefficient@{k}"] = gini_coefficient(all_preds, k=k)

            results[f"intra_list_diversity@{k}"] = intra_list_diversity(
                all_preds, item_embeddings=self.item_embeddings, k=k
            )
            results[f"prediction_entropy@{k}"] = prediction_entropy(all_preds, k=k)

            if self.item_popularities:
                results[f"novelty@{k}"] = novelty_at_k(all_preds, self.item_popularities, k=k)

        return results

    @classmethod
    def compare_models_on_slice(
        cls,
        models_predictions: Dict[str, Sequence[Sequence[Any]]],
        actuals: Sequence[Sequence[Any]],
        contexts: Sequence[Dict[str, Any]],
        slice_feature: str,
        k: int = 10,
        catalog_items: Optional[Sequence[Any]] = None,
    ) -> pd.DataFrame:
        """
        Direct head-to-head comparison of multiple models broken down across context slices.

        Args:
            models_predictions: Dict mapping model_name -> list of prediction sequences.
            actuals: List of ground-truth sequences.
            contexts: List of context dictionaries for each session.
            slice_feature: Context feature to group by.
            k: Evaluation cutoff K.
            catalog_items: Optional catalog list.

        Returns:
            pd.DataFrame comparing models side-by-side across slices.
        """
        rows = []
        for model_name, preds in models_predictions.items():
            evaluator = cls(catalog_items=catalog_items)
            for i in range(len(actuals)):
                evaluator.add_session(
                    session_id=i,
                    actual=actuals[i],
                    predicted=preds[i],
                    context=contexts[i],
                )
            slice_df = evaluator.evaluate_slice(slice_feature, k_values=[k])
            for _, r in slice_df.iterrows():
                row_dict = r.to_dict()
                row_dict["model"] = model_name
                rows.append(row_dict)

        comp_df = pd.DataFrame(rows)
        if not comp_df.empty:
            cols = ["model", "slice_feature", "slice_value", "session_count", f"recall@{k}", f"ndcg@{k}", f"mrr@{k}", f"hit_rate@{k}", f"coverage@{k}", f"ild@{k}"]
            avail_cols = [c for c in cols if c in comp_df.columns]
            comp_df = comp_df[avail_cols].sort_values(by=["slice_value", f"ndcg@{k}"], ascending=[True, False]).reset_index(drop=True)
        return comp_df


# =====================================================================
# Standalone Demonstration & Verification
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 80)
    print("DAY 5: CONTEXT-SLICE & RECOMMENDATION DIVERSITY / COVERAGE BENCHMARK")
    print("=" * 80)

    # 1. Synthesize catalog and contextual test traffic
    np.random.seed(42)
    catalog = [f"item_{i:04d}" for i in range(1000)]
    # Item embeddings (32D)
    item_embs = {item: np.random.randn(32).astype(np.float32) for item in catalog}
    # Power-law item popularity
    raw_pops = np.random.pareto(a=1.5, size=len(catalog)) + 1.0
    total_pops = np.sum(raw_pops)
    item_pops = {item: float(raw_pops[i] / total_pops) for i, item in enumerate(catalog)}

    # Generate 300 test sessions with seasonal and weekend context
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    channels = [1, 2]  # Store vs Online
    n_sessions = 300

    evaluator = SliceEvaluator(
        catalog_items=catalog,
        item_embeddings=item_embs,
        item_popularities=item_pops,
    )

    print(f"\n[1] Generated Catalog & Evaluation Sessions:")
    print(f"    - Total Catalog Items    : {len(catalog)}")
    print(f"    - Evaluation Sessions    : {n_sessions}")
    print(f"    - Context Features       : season, is_weekend, sales_channel_id")

    # Simulate recommendations with context-dependent hit rates:
    # Model performs well in Winter/Summer, lower in Spring/Fall
    for s_idx in range(n_sessions):
        season_val = np.random.choice(seasons, p=[0.35, 0.20, 0.30, 0.15])
        is_wknd = int(np.random.rand() > 0.6)
        channel = int(np.random.choice(channels))

        ctx = {
            "season": season_val,
            "is_weekend": is_wknd,
            "sales_channel_id": channel,
        }

        # Select true item
        actual_item = np.random.choice(catalog)
        # Prediction list: higher hit rate in Winter / Summer
        hit_prob = 0.70 if season_val in ["Winter", "Summer"] else 0.30
        if np.random.rand() < hit_prob:
            predicted_items = [actual_item] + list(np.random.choice(catalog, size=19, replace=False))
        else:
            other_items = [it for it in catalog if it != actual_item]
            predicted_items = list(np.random.choice(other_items, size=20, replace=False))

        evaluator.add_session(
            session_id=f"sess_{s_idx:04d}",
            actual=[actual_item],
            predicted=predicted_items,
            context=ctx,
        )

    # 2. Evaluate Seasonal Context Slice
    print("\n[2] Seasonal Context Slice Evaluation (K=10):")
    season_slice_df = evaluator.evaluate_slice("season", k_values=[5, 10])
    print(season_slice_df.to_string(index=False))

    # 3. Disparity Analysis
    print("\n[3] Contextual Disparity Analysis (NDCG@10):")
    disparity = evaluator.compute_slice_disparity(season_slice_df, metric="ndcg@10")
    print(f"    - Metric Analyzed     : {disparity['metric']}")
    print(f"    - Best Slice          : {disparity['max_slice']} ({disparity['max_value']:.4f})")
    print(f"    - Worst Slice         : {disparity['min_slice']} ({disparity['min_value']:.4f})")
    print(f"    - Disparity Ratio     : {disparity['disparity_ratio']:.4f}")
    print(f"    - Disparity Flagged?  : {disparity['disparity_flagged']}")

    # 4. Global Beyond-Accuracy Metrics
    print("\n[4] Catalog-Wide Beyond-Accuracy Metrics:")
    beyond_metrics = evaluator.evaluate_global_beyond_accuracy(k_values=[5, 10])
    for m_name, m_val in beyond_metrics.items():
        print(f"    - {m_name:<28}: {m_val:.4f}")

    print("\n" + "=" * 80)
    print("Day 5 Context-Slice & Diversity/Coverage Benchmark Completed Successfully!")
    print("=" * 80 + "\n")
