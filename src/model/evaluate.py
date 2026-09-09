"""
Day 6: Model Evaluation Pipeline — End-to-End Retrieval Evaluation from Embeddings.

ModelEvaluationPipeline loads the saved embedding artefacts from train.py,
builds an ExactSearchIndex (and optionally an IVFIndex), then runs the
Day 2 RetrievalEvaluator to compute Recall@K, NDCG@K, MRR@K, and Hit Rate@K
on a held-out test split. It also generates a per-user diagnostic report.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.ann_search import ExactSearchIndex, IVFIndex, ANNEvaluator
from src.evaluation.evaluator import EvaluationDataset, RetrievalEvaluator

logger = logging.getLogger(__name__)


class ModelEvaluationPipeline:
    """
    End-to-end evaluation pipeline for the Two-Tower Recommendation Model.

    Steps:
        1. Load saved user/item embedding arrays from disk (output of ModelTrainer)
        2. Build an ExactSearchIndex over item embeddings
        3. Build an EvaluationDataset from the test interaction split
        4. Run RetrievalEvaluator.evaluate_embeddings() for Recall@K, NDCG@K, MRR@K
        5. Optionally compare vs. IVFIndex (ANN approximation) for latency benchmarking
        6. Output a per-user diagnostic DataFrame and summary metrics dict
    """

    def __init__(
        self,
        model_dir: str = "data/processed/model",
        k_values: Tuple[int, ...] = (5, 10, 20),
    ):
        """
        Initialise the evaluation pipeline.

        Args:
            model_dir: Directory containing user_ids.npy, user_embeddings.npy,
                       item_ids.npy, item_embeddings.npy.
            k_values: Sequence of K cutoffs for evaluation metrics.
        """
        self.model_dir = Path(model_dir)
        self.k_values = k_values

        self.user_ids: Optional[np.ndarray] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None

        self.exact_index: Optional[ExactSearchIndex] = None
        self.ivf_index: Optional[IVFIndex] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Artefact loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_embeddings(self) -> "ModelEvaluationPipeline":
        """
        Load user and item embedding arrays from the model output directory.

        Returns:
            self (for method chaining).

        Raises:
            FileNotFoundError: If any required .npy file is missing.
        """
        required = ["user_ids", "user_embeddings", "item_ids", "item_embeddings"]
        for key in required:
            fpath = self.model_dir / f"{key}.npy"
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Embedding file not found: {fpath}\n"
                    "Run src/model/train.py first to generate embeddings."
                )

        self.user_ids       = np.load(str(self.model_dir / "user_ids.npy"),       allow_pickle=True)
        self.user_embeddings = np.load(str(self.model_dir / "user_embeddings.npy"))
        self.item_ids       = np.load(str(self.model_dir / "item_ids.npy"),       allow_pickle=True)
        self.item_embeddings = np.load(str(self.model_dir / "item_embeddings.npy"))

        logger.info(
            "Loaded embeddings — users: %s, items: %s",
            self.user_embeddings.shape, self.item_embeddings.shape,
        )
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # Index construction
    # ─────────────────────────────────────────────────────────────────────────

    def build_exact_index(self) -> ExactSearchIndex:
        """
        Build an ExactSearchIndex (exhaustive brute-force) over item embeddings.

        Returns:
            Fitted ExactSearchIndex.
        """
        if self.item_embeddings is None:
            raise RuntimeError("Call load_embeddings() first.")

        self.exact_index = ExactSearchIndex(normalize=True)
        self.exact_index.fit(self.item_ids, self.item_embeddings)
        logger.info("ExactSearchIndex built with %d items.", self.exact_index.num_items)
        return self.exact_index

    def build_ivf_index(self, nlist: int = 32, nprobe: int = 4) -> IVFIndex:
        """
        Build an IVFIndex (approximate nearest neighbour) over item embeddings.

        Args:
            nlist: Number of Voronoi clusters.
            nprobe: Number of clusters to probe during search.

        Returns:
            Fitted IVFIndex.
        """
        if self.item_embeddings is None:
            raise RuntimeError("Call load_embeddings() first.")

        self.ivf_index = IVFIndex(nlist=nlist, nprobe=nprobe, normalize=True, random_state=42)
        self.ivf_index.fit(self.item_ids, self.item_embeddings)
        logger.info("IVFIndex built — nlist=%d, nprobe=%d.", nlist, nprobe)
        return self.ivf_index

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_df: pd.DataFrame,
        user_col: str = "customer_id",
        item_col: str = "article_id",
        time_col: str = "t_dat",
        train_history: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality of the Two-Tower model on a test interaction split.

        Args:
            test_df: Test interaction DataFrame (must contain user_col and item_col).
            user_col: Column name for user identifiers.
            item_col: Column name for item identifiers.
            time_col: Optional time column (used if present, otherwise ignored).
            train_history: Optional dict of {user_id → set of train items} to mask
                           out previously seen items from the retrieval ranking.

        Returns:
            Dict[str, float]: Aggregated evaluation metrics (Recall, NDCG, MRR, HitRate @ K).
        """
        if self.user_embeddings is None or self.exact_index is None:
            raise RuntimeError("Call load_embeddings() and build_exact_index() first.")

        # Build EvaluationDataset from test split
        tc = time_col if time_col in test_df.columns else None
        eval_dataset = EvaluationDataset(
            test_df, user_col=user_col, item_col=item_col, time_col=tc
        )
        ground_truth = eval_dataset.ground_truth
        test_users = eval_dataset.get_test_users()

        logger.info("Evaluating on %d test users.", len(test_users))

        # Filter user embeddings to those present in the test set
        user_id_str = [str(u) for u in self.user_ids]
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_id_str)}

        eval_user_ids = [u for u in test_users if u in user_id_to_idx]
        if not eval_user_ids:
            logger.warning("No overlap between test users and embedding users.")
            return {}

        eval_indices = [user_id_to_idx[u] for u in eval_user_ids]
        eval_user_embs = self.user_embeddings[eval_indices]

        # Run retrieval evaluation
        evaluator = RetrievalEvaluator(k_values=self.k_values)
        metrics = evaluator.evaluate_embeddings(
            user_embeddings=eval_user_embs,
            item_embeddings=self.item_embeddings,
            user_ids=eval_user_ids,
            item_ids=[str(i) for i in self.item_ids],
            ground_truth=ground_truth,
            filter_train_history=train_history,
        )
        return metrics

    def generate_diagnostic_report(
        self,
        test_df: pd.DataFrame,
        k_eval: int = 10,
        user_col: str = "customer_id",
        item_col: str = "article_id",
        time_col: str = "t_dat",
    ) -> pd.DataFrame:
        """
        Generate a per-user diagnostic report with recall, NDCG, MRR, and hit metrics.

        Args:
            test_df: Test interaction DataFrame.
            k_eval: K cutoff for the per-user report.
            user_col: User column name.
            item_col: Item column name.
            time_col: Optional time column name.

        Returns:
            pd.DataFrame with columns: user_id, num_ground_truth, recall@k, ndcg@k, etc.
        """
        if self.user_embeddings is None or self.exact_index is None:
            raise RuntimeError("Call load_embeddings() and build_exact_index() first.")

        tc = time_col if time_col in test_df.columns else None
        eval_dataset = EvaluationDataset(
            test_df, user_col=user_col, item_col=item_col, time_col=tc
        )
        ground_truth = eval_dataset.ground_truth
        test_users = eval_dataset.get_test_users()

        user_id_str = [str(u) for u in self.user_ids]
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_id_str)}

        eval_user_ids = [u for u in test_users if u in user_id_to_idx]
        eval_indices  = [user_id_to_idx[u] for u in eval_user_ids]
        eval_user_embs = self.user_embeddings[eval_indices]

        # Retrieve top-k predictions for each eval user
        item_id_str = [str(i) for i in self.item_ids]
        item_id_to_idx = {iid: idx for idx, iid in enumerate(item_id_str)}
        item_embs_norm = self.item_embeddings / np.maximum(
            np.linalg.norm(self.item_embeddings, axis=1, keepdims=True), 1e-12
        )
        user_embs_norm = eval_user_embs / np.maximum(
            np.linalg.norm(eval_user_embs, axis=1, keepdims=True), 1e-12
        )

        scores = np.matmul(user_embs_norm, item_embs_norm.T)   # (eval_users, items)
        predictions = {}
        for i, uid in enumerate(eval_user_ids):
            top_indices = np.argsort(-scores[i])[:k_eval]
            predictions[uid] = [item_id_str[j] for j in top_indices]

        # Build per-user report via RetrievalEvaluator
        evaluator = RetrievalEvaluator(k_values=[k_eval])
        report_df = evaluator.generate_evaluation_report_df(
            ground_truth={u: ground_truth[u] for u in eval_user_ids if u in ground_truth},
            predictions=predictions,
            k_eval=k_eval,
        )
        return report_df

    # ─────────────────────────────────────────────────────────────────────────
    # ANN vs. Exact comparison
    # ─────────────────────────────────────────────────────────────────────────

    def benchmark_ann_vs_exact(
        self,
        num_queries: int = 100,
        k: int = 10,
        nprobe_values: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Benchmark IVF approximate retrieval vs. exact search.

        Args:
            num_queries: Number of random query embeddings to benchmark.
            k: Top-K retrieval cutoff.
            nprobe_values: List of nprobe values to sweep.

        Returns:
            pd.DataFrame from ANNEvaluator.benchmark_tradeoff_sweep.
        """
        if self.exact_index is None or self.ivf_index is None:
            raise RuntimeError("Build both exact_index and ivf_index first.")

        np.random.seed(42)
        query_embs = np.random.randn(num_queries, self.item_embeddings.shape[1]).astype(np.float32)

        df = ANNEvaluator.benchmark_tradeoff_sweep(
            ann_index=self.ivf_index,
            exact_index=self.exact_index,
            queries=query_embs,
            k=k,
            nprobe_list=nprobe_values,
            benchmark_runs=3,
        )
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo — uses synthetic embeddings if no trained model exists
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 70)
    print("DAY 6: MODEL EVALUATION PIPELINE DEMO")
    print("=" * 70)

    import tempfile, shutil

    np.random.seed(42)

    # Create a temp model dir with synthetic embeddings
    tmp_dir = tempfile.mkdtemp(prefix="ttmodel_")
    try:
        NUM_USERS  = 200
        NUM_ITEMS  = 500
        EMB_DIM    = 64

        print(f"\n[1] Generating synthetic embeddings ({NUM_USERS} users, {NUM_ITEMS} items)...")
        u_ids  = np.array([f"user_{i}" for i in range(NUM_USERS)])
        i_ids  = np.array([f"item_{i}" for i in range(NUM_ITEMS)])
        u_embs = np.random.randn(NUM_USERS, EMB_DIM).astype(np.float32)
        i_embs = np.random.randn(NUM_ITEMS, EMB_DIM).astype(np.float32)
        # L2-normalise
        u_embs /= np.linalg.norm(u_embs, axis=1, keepdims=True)
        i_embs /= np.linalg.norm(i_embs, axis=1, keepdims=True)

        np.save(os.path.join(tmp_dir, "user_ids.npy"),        u_ids)
        np.save(os.path.join(tmp_dir, "user_embeddings.npy"), u_embs)
        np.save(os.path.join(tmp_dir, "item_ids.npy"),        i_ids)
        np.save(os.path.join(tmp_dir, "item_embeddings.npy"), i_embs)

        print("\n[2] Building evaluation pipeline...")
        pipeline = ModelEvaluationPipeline(model_dir=tmp_dir, k_values=(5, 10, 20))
        pipeline.load_embeddings()
        pipeline.build_exact_index()
        pipeline.build_ivf_index(nlist=16, nprobe=4)

        # Synthetic test interactions — assign positive item as the nearest neighbour
        # for each user to ensure non-zero recall
        print("\n[3] Creating synthetic test interactions...")
        scores_mat = np.matmul(u_embs, i_embs.T)
        test_records = []
        for u_idx in range(min(NUM_USERS, 50)):
            nearest_item_idx = int(np.argmax(scores_mat[u_idx]))
            test_records.append({
                "customer_id": f"user_{u_idx}",
                "article_id":  f"item_{nearest_item_idx}",
                "t_dat":       "2026-08-01",
            })
        test_df = pd.DataFrame(test_records)

        print("\n[4] Running retrieval evaluation...")
        metrics = pipeline.evaluate(test_df)
        print("\n    --- Retrieval Metrics ---")
        for metric, val in sorted(metrics.items()):
            print(f"    {metric:<18} : {val:.4f}")

        print("\n[5] Per-user diagnostic report (top 5 rows):")
        report = pipeline.generate_diagnostic_report(test_df, k_eval=10)
        print(report.head(5).to_string(index=False))

        print("\n[6] ANN vs. Exact latency/recall trade-off sweep:")
        sweep_df = pipeline.benchmark_ann_vs_exact(num_queries=50, k=10)
        print(sweep_df.to_string(index=False))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Day 6 Model Evaluation Pipeline Demo Completed Successfully!")
    print("=" * 70 + "\n")
