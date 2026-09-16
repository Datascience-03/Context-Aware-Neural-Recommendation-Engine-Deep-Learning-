
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Make the project root importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Optional Airflow imports
# ─────────────────────────────────────────────────────────────────────────────
#
# Airflow 3.x:
#   - DAG is imported from airflow.sdk
#   - PythonOperator is provided by the standard provider
#   - days_ago() is no longer used
#
# Native Windows cannot run Airflow itself; Airflow should be executed through
# WSL2/Linux or Linux containers. The ML task functions below remain usable
# directly on Windows.
# ─────────────────────────────────────────────────────────────────────────────
try:
    from airflow.sdk import DAG
    from airflow.providers.standard.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
    AIRFLOW_IMPORT_ERROR = None

except (ImportError, ModuleNotFoundError, AttributeError) as exc:
    AIRFLOW_AVAILABLE = False
    AIRFLOW_IMPORT_ERROR = exc
    DAG = None  # type: ignore[assignment,misc]
    PythonOperator = None  # type: ignore[assignment,misc]


# ─────────────────────────────────────────────────────────────────────────────
# Default task configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "recommendation-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

_BASE_PATH = os.getenv("DATA_BASE_PATH", "data/reduced")
_PROCESSED_PATH = os.getenv("DATA_PROC_PATH", "data/processed")
_MODEL_DIR = os.getenv("MODEL_DIR", "data/processed/model")
_EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "64"))
_BATCH_SIZE = int(os.getenv("TRAIN_BATCH", "256"))
_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "15"))
_TOP_K_EVAL = int(os.getenv("TOP_K_EVAL", "10"))


# ─────────────────────────────────────────────────────────────────────────────
# Task callables — pure Python functions, no Airflow imports inside
# ─────────────────────────────────────────────────────────────────────────────

def task_generate_data(**context: Any) -> None:
    """Task 1: Ensure reduced dataset CSVs exist."""
    import subprocess

    transactions_path = Path(_BASE_PATH) / "transactions_reduced.csv"

    if transactions_path.exists():
        logger.info(
            "Reduced dataset already exists at %s. Skipping generation.",
            _BASE_PATH,
        )
        return

    logger.info("Reduced dataset not found. Running generate_data.py ...")

    result = subprocess.run(
        [sys.executable, "generate_data.py"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        raise RuntimeError(f"generate_data.py failed:\n{result.stderr}")

    logger.info("Data generation complete.")


def task_run_spark_pipeline(**context: Any) -> None:
    """Task 2: Clean raw data with PySpark/Pandas fallback."""
    from src.pipeline.pyspark_pipeline import main as run_pyspark

    os.makedirs(_PROCESSED_PATH, exist_ok=True)
    run_pyspark()

    logger.info(
        "Spark/Pandas pipeline completed — processed files at %s",
        _PROCESSED_PATH,
    )


def task_extract_features(**context: Any) -> None:
    """Task 3: Extract contextual/recency features and fit vocabulary indices."""
    from src.pipeline.integrate import run_integration_pipeline

    run_integration_pipeline()
    logger.info("Feature extraction and vocabulary fitting complete.")


def task_train_model(**context: Any) -> Dict[str, float]:
    """
    Task 4: Train the Two-Tower model on the integrated training dataset.

    Returns:
        Final training loss and top-1 accuracy for XCom.
    """
    import numpy as np
    import pandas as pd

    from src.model.train import ModelTrainer

    training_csv = Path(_PROCESSED_PATH) / "final_training_data.csv"

    if not training_csv.exists():
        logger.warning(
            "final_training_data.csv not found. Generating synthetic data for demo."
        )
        from src.model.train import _make_synthetic_df

        np.random.seed(42)
        df = _make_synthetic_df(
            n_rows=10_000,
            num_users=1000,
            num_items=2000,
        )
        num_users = 1000
        num_items = 2000
        num_pg = 20
        num_cg = 50
    else:
        df = pd.read_csv(training_csv)

        num_users = (
            int(df["customer_id_idx"].max()) + 1
            if "customer_id_idx" in df.columns
            else 1000
        )
        num_items = (
            int(df["article_id_idx"].max()) + 1
            if "article_id_idx" in df.columns
            else 2000
        )
        num_pg = (
            int(df["product_group_name_idx"].max()) + 1
            if "product_group_name_idx" in df.columns
            else 20
        )
        num_cg = (
            int(df["colour_group_name_idx"].max()) + 1
            if "colour_group_name_idx" in df.columns
            else 50
        )

    split = int(0.90 * len(df))
    df_train = df.iloc[:split]
    df_val = df.iloc[split:]

    trainer = ModelTrainer(
        embedding_dim=_EMBEDDING_DIM,
        batch_size=_BATCH_SIZE,
        epochs=_EPOCHS,
        learning_rate=1e-3,
        patience=3,
        output_dir=_MODEL_DIR,
    )

    trainer.build_model(
        num_users=num_users,
        num_items=num_items,
        num_product_groups=num_pg,
        num_colour_groups=num_cg,
    )

    train_ds = trainer.build_tf_dataset(df_train, shuffle=True)
    val_ds = trainer.build_tf_dataset(df_val, shuffle=False)

    history = trainer.train(train_ds, val_ds)

    final_loss = float(history.history["loss"][-1])
    final_acc = float(history.history.get("top1_accuracy", [0.0])[-1])

    trainer.save_model_weights()

    logger.info(
        "Training complete — final_loss=%.4f, top1_acc=%.4f",
        final_loss,
        final_acc,
    )

    return {
        "final_loss": final_loss,
        "top1_accuracy": final_acc,
    }


def task_export_embeddings(**context: Any) -> None:
    """Task 5: Export user and item embedding matrices as .npy files."""
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from src.model.train import ModelTrainer

    training_csv = Path(_PROCESSED_PATH) / "final_training_data.csv"

    if not training_csv.exists():
        from src.model.train import _make_synthetic_df

        df = _make_synthetic_df(
            n_rows=5_000,
            num_users=500,
            num_items=1000,
        )
    else:
        df = pd.read_csv(training_csv)

    num_users = (
        int(df["customer_id_idx"].max()) + 1
        if "customer_id_idx" in df.columns
        else 500
    )
    num_items = (
        int(df["article_id_idx"].max()) + 1
        if "article_id_idx" in df.columns
        else 1000
    )
    num_pg = (
        int(df["product_group_name_idx"].max()) + 1
        if "product_group_name_idx" in df.columns
        else 20
    )
    num_cg = (
        int(df["colour_group_name_idx"].max()) + 1
        if "colour_group_name_idx" in df.columns
        else 50
    )

    trainer = ModelTrainer(
        embedding_dim=_EMBEDDING_DIM,
        output_dir=_MODEL_DIR,
    )

    trainer.build_model(
        num_users=num_users,
        num_items=num_items,
        num_product_groups=num_pg,
        num_colour_groups=num_cg,
    )

    weights_path = Path(_MODEL_DIR) / "final_weights.weights.h5"

    if weights_path.exists():
        dummy_batch = {
            "customer_id_idx": np.zeros((1,), dtype=np.int32),
            "month_sin": np.zeros((1,), dtype=np.float32),
            "month_cos": np.zeros((1,), dtype=np.float32),
            "day_of_week_sin": np.zeros((1,), dtype=np.float32),
            "day_of_week_cos": np.zeros((1,), dtype=np.float32),
            "is_weekend": np.zeros((1,), dtype=np.float32),
            "days_since_last_purchase": np.zeros((1,), dtype=np.float32),
            "purchase_sequence": np.zeros((1,), dtype=np.float32),
            "article_id_idx": np.zeros((1,), dtype=np.int32),
            "product_group_name_idx": np.zeros((1,), dtype=np.int32),
            "colour_group_name_idx": np.zeros((1,), dtype=np.int32),
            "popularity_over_time": np.zeros((1,), dtype=np.float32),
        }

        trainer.model(
            {k: tf.constant(v) for k, v in dummy_batch.items()},
            training=False,
        )

        trainer.model.load_weights(str(weights_path))
        logger.info("Loaded model weights from %s", weights_path)
    else:
        logger.warning(
            "Weights file not found at %s. Exporting embeddings from the "
            "freshly initialised model.",
            weights_path,
        )

    user_feat_df = df.drop_duplicates(
        subset="customer_id_idx"
    ).reset_index(drop=True)

    item_feat_df = df.drop_duplicates(
        subset="article_id_idx"
    ).reset_index(drop=True)

    user_ids_arr = user_feat_df["customer_id_idx"].astype(str).values
    item_ids_arr = item_feat_df["article_id_idx"].astype(str).values

    trainer.save_user_embeddings(user_ids_arr, user_feat_df)
    trainer.save_item_embeddings(item_ids_arr, item_feat_df)

    logger.info("Embedding export complete.")


def task_build_ann_index(**context: Any) -> None:
    """
    Task 6: Build and validate an ANN retrieval index from item embeddings.

    The serving API rebuilds the index in memory at startup; this task validates
    that the exported embeddings are compatible with the retrieval layer.
    """
    import numpy as np

    from src.retrieval.ann_search import ExactSearchIndex, IVFIndex

    item_embs_path = Path(_MODEL_DIR) / "item_embeddings.npy"
    item_ids_path = Path(_MODEL_DIR) / "item_ids.npy"

    if not item_embs_path.exists():
        raise FileNotFoundError(
            f"item_embeddings.npy not found at {_MODEL_DIR}. "
            "Run task_export_embeddings first."
        )

    if not item_ids_path.exists():
        raise FileNotFoundError(
            f"item_ids.npy not found at {_MODEL_DIR}. "
            "Run task_export_embeddings first."
        )

    item_embeddings = np.load(str(item_embs_path))
    item_ids = np.load(str(item_ids_path), allow_pickle=True)

    n_items = len(item_ids)

    if n_items == 0:
        raise ValueError("No item embeddings were exported.")

    nlist = max(1, min(64, n_items // 10))

    if n_items >= nlist * 2:
        logger.info(
            "Building IVFIndex (nlist=%d) for %d items.",
            nlist,
            n_items,
        )
        index = IVFIndex(
            nlist=nlist,
            nprobe=min(4, nlist),
            normalize=True,
            random_state=42,
        )
    else:
        logger.info(
            "Building ExactSearchIndex for %d items.",
            n_items,
        )
        index = ExactSearchIndex(normalize=True)

    index.fit(item_ids, item_embeddings)

    query = item_embeddings[:1]
    retrieved_ids, scores = index.search(
        query,
        k=min(5, n_items),
    )

    logger.info(
        "ANN index sanity check: top ids=%s, scores=%s",
        retrieved_ids[0][:5],
        scores[0][:5].round(4),
    )
    logger.info("ANN index build and validation complete.")


def task_run_evaluation(**context: Any) -> None:
    """
    Task 7: Run RetrievalEvaluator and save metrics to JSON.

    Evaluation uses the encoded customer/article IDs that correspond to the
    saved embedding matrices.
    """
    import numpy as np
    import pandas as pd

    from src.model.evaluate import ModelEvaluationPipeline

    pipeline = ModelEvaluationPipeline(
        model_dir=_MODEL_DIR,
        k_values=(5, 10, 20),
    )

    try:
        pipeline.load_embeddings()
        pipeline.build_exact_index()
    except FileNotFoundError as exc:
        logger.warning(
            "Embedding files not found: %s. Skipping evaluation.",
            exc,
        )
        return

    training_csv = Path(_PROCESSED_PATH) / "final_training_data.csv"

    if training_csv.exists():
        df = pd.read_csv(training_csv)
        test_df = df.tail(max(100, len(df) // 10)).copy()

        if "customer_id_idx" not in test_df.columns:
            raise ValueError(
                "customer_id_idx is required for embedding-based evaluation."
            )

        if "article_id_idx" not in test_df.columns:
            raise ValueError(
                "article_id_idx is required for embedding-based evaluation."
            )

        test_df["customer_id_eval"] = (
            test_df["customer_id_idx"].astype(str)
        )
        test_df["article_id_eval"] = (
            test_df["article_id_idx"].astype(str)
        )

        user_col = "customer_id_eval"
        item_col = "article_id_eval"

    else:
        logger.warning(
            "No training CSV found. Using synthetic test interactions."
        )

        n = len(pipeline.user_ids)

        scores_mat = np.matmul(
            pipeline.user_embeddings,
            pipeline.item_embeddings.T,
        )

        records = []

        for u_idx in range(min(n, 200)):
            nearest = int(np.argmax(scores_mat[u_idx]))

            records.append(
                {
                    "customer_id": str(pipeline.user_ids[u_idx]),
                    "article_id": str(pipeline.item_ids[nearest]),
                }
            )

        test_df = pd.DataFrame(records)
        user_col = "customer_id"
        item_col = "article_id"

    metrics = pipeline.evaluate(
        test_df,
        user_col=user_col,
        item_col=item_col,
    )

    if metrics:
        os.makedirs(_MODEL_DIR, exist_ok=True)

        metrics_path = Path(_MODEL_DIR) / "eval_metrics.json"

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(
            "Evaluation complete. Metrics saved to %s",
            metrics_path,
        )

        for metric, val in sorted(metrics.items()):
            logger.info(
                "    %-20s : %.4f",
                metric,
                val,
            )
    else:
        logger.warning("No evaluation metrics returned.")


def task_notify_completion(**context: Any) -> None:
    """Task 8: Log pipeline completion summary."""
    run_id = context.get("run_id", "unknown")

    dag_context = context.get("dag")
    dag_id = (
        dag_context.dag_id
        if dag_context is not None
        else "recommendation_training_pipeline"
    )

    metrics_path = Path(_MODEL_DIR) / "eval_metrics.json"

    summary_lines = [
        f"  DAG            : {dag_id}",
        f"  Run ID         : {run_id}",
        (
            "  Timestamp      : "
            f"{datetime.now(timezone.utc).isoformat()}"
        ),
        f"  Model dir      : {_MODEL_DIR}",
    ]

    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

        summary_lines += [
            f"  {key:<20} : {value:.4f}"
            for key, value in sorted(metrics.items())
        ]

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETION NOTIFICATION")

    for line in summary_lines:
        logger.info(line)

    logger.info("=" * 60)

    # TODO: extend with Slack/email/PagerDuty notification.


# ─────────────────────────────────────────────────────────────────────────────
# DAG task definitions
# ─────────────────────────────────────────────────────────────────────────────

_TASK_DEFINITIONS = [
    {
        "task_id": "generate_data",
        "callable": task_generate_data,
        "doc": "Ensure reduced dataset CSVs exist; generate if missing.",
        "upstream": [],
    },
    {
        "task_id": "run_spark_pipeline",
        "callable": task_run_spark_pipeline,
        "doc": "Clean and transform raw data with PySpark/Pandas; output parquet.",
        "upstream": ["generate_data"],
    },
    {
        "task_id": "extract_features",
        "callable": task_extract_features,
        "doc": "Extract contextual + recency features; build vocabulary indices.",
        "upstream": ["run_spark_pipeline"],
    },
    {
        "task_id": "train_model",
        "callable": task_train_model,
        "doc": "Train Two-Tower model on integrated features; save weights.",
        "upstream": ["extract_features"],
    },
    {
        "task_id": "export_embeddings",
        "callable": task_export_embeddings,
        "doc": "Export user/item embedding matrices as .npy for ANN indexing.",
        "upstream": ["train_model"],
    },
    {
        "task_id": "build_ann_index",
        "callable": task_build_ann_index,
        "doc": "Build and validate the ANN retrieval index from item embeddings.",
        "upstream": ["export_embeddings"],
    },
    {
        "task_id": "run_evaluation",
        "callable": task_run_evaluation,
        "doc": "Evaluate Recall@K/NDCG@K on test split; persist metrics.json.",
        "upstream": ["build_ann_index"],
    },
    {
        "task_id": "notify_completion",
        "callable": task_notify_completion,
        "doc": "Log (and optionally notify) pipeline completion summary.",
        "upstream": ["run_evaluation"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Airflow DAG definition
# ─────────────────────────────────────────────────────────────────────────────

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="recommendation_training_pipeline",
        default_args=_DEFAULT_ARGS,
        description=(
            "Daily two-tower recommendation model training pipeline: "
            "data ingestion → feature engineering → model training → "
            "embedding export → ANN index build → evaluation → notification."
        ),
        schedule="@daily",
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        catchup=False,
        max_active_runs=1,
        tags=["recommendation", "two-tower", "deep-learning"],
    ) as dag:

        airflow_tasks: Dict[str, Any] = {}

        for task_def in _TASK_DEFINITIONS:
            task = PythonOperator(
                task_id=task_def["task_id"],
                python_callable=task_def["callable"],
                doc_md=task_def["doc"],
            )
            airflow_tasks[task_def["task_id"]] = task

        for task_def in _TASK_DEFINITIONS:
            for upstream_id in task_def["upstream"]:
                airflow_tasks[upstream_id] >> airflow_tasks[
                    task_def["task_id"]
                ]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n" + "=" * 70)
    print("DAY 7: AIRFLOW DAG — recommendation_training_pipeline")
    print("=" * 70)

    if AIRFLOW_AVAILABLE:
        print("\nAirflow imports available. DAG registered successfully.")
    else:
        print("\nAirflow DAG runtime is unavailable in this environment.")
        if AIRFLOW_IMPORT_ERROR is not None:
            print(
                f"Reason: {type(AIRFLOW_IMPORT_ERROR).__name__}: "
                f"{AIRFLOW_IMPORT_ERROR}"
            )
        print(
            "The standalone task graph and ML task functions remain available."
        )

    print("\nTask Graph (sequential pipeline):")

    for i, task_def in enumerate(_TASK_DEFINITIONS):
        arrow = "→ " if i > 0 else "  "
        print(
            f"  {arrow}[{i + 1}] "
            f"{task_def['task_id']:<25} "
            f"{task_def['doc']}"
        )

    print("\n" + "-" * 70)
    print("Dry-run: validating ANN index and completion notification...")

    import shutil
    import tempfile

    import numpy as np

    tmp_dir = tempfile.mkdtemp(prefix="dag_dryrun_")

    try:
        os.makedirs(tmp_dir, exist_ok=True)

        np.random.seed(0)

        n_items, embedding_dim = 200, 64

        item_embeddings = np.random.randn(
            n_items,
            embedding_dim,
        ).astype(np.float32)

        item_embeddings /= np.linalg.norm(
            item_embeddings,
            axis=1,
            keepdims=True,
        )

        item_ids = np.array(
            [f"item_{i}" for i in range(n_items)]
        )

        np.save(
            os.path.join(tmp_dir, "item_embeddings.npy"),
            item_embeddings,
        )
        np.save(
            os.path.join(tmp_dir, "item_ids.npy"),
            item_ids,
        )

        original_model_dir = _MODEL_DIR

        try:
            _MODEL_DIR = tmp_dir

            task_build_ann_index()
            task_notify_completion(run_id="dry_run_001")

        finally:
            _MODEL_DIR = original_model_dir

        print("\nDry-run completed successfully.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Day 7 Airflow DAG demo completed!")
    print("=" * 70 + "\n")
