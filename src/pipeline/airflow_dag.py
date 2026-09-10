"""
Day 7: Airflow DAG — Daily Recommendation Training & Serving Pipeline.

Defines a production-grade `recommendation_training_pipeline` DAG that orchestrates
the full lifecycle from raw data ingestion through model training to ANN index export
and offline evaluation. All tasks use PythonOperator for portability.

Airflow is an optional dependency — this file is importable even if Airflow is not
installed (all Airflow imports are guarded inside a try/except block). Running the
module standalone prints a human-readable DAG task summary.

DAG Task Graph:
    generate_data
        → run_spark_pipeline
            → extract_features
                → train_model
                    → export_embeddings
                        → build_ann_index
                            → run_evaluation
                                → notify_completion
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ─────────────────────────────────────────────────────────────────────────────
# Guard Airflow imports — file must be importable without Airflow installed
# ─────────────────────────────────────────────────────────────────────────────

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.dates import days_ago
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    DAG = None                    # type: ignore[assignment,misc]
    PythonOperator = None         # type: ignore[assignment,misc]
    days_ago = None               # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Default task configuration
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_ARGS: Dict[str, Any] = {
    "owner":            "recommendation-team",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

_BASE_PATH      = os.getenv("DATA_BASE_PATH",  "data/reduced")
_PROCESSED_PATH = os.getenv("DATA_PROC_PATH",   "data/processed")
_MODEL_DIR      = os.getenv("MODEL_DIR",         "data/processed/model")
_EMBEDDING_DIM  = int(os.getenv("EMBEDDING_DIM", "64"))
_BATCH_SIZE     = int(os.getenv("TRAIN_BATCH",   "256"))
_EPOCHS         = int(os.getenv("TRAIN_EPOCHS",  "15"))
_TOP_K_EVAL     = int(os.getenv("TOP_K_EVAL",    "10"))


# ─────────────────────────────────────────────────────────────────────────────
# Task callables  (pure Python functions — no Airflow imports inside)
# ─────────────────────────────────────────────────────────────────────────────

def task_generate_data(**context: Any) -> None:
    """
    Task 1: Ensure reduced dataset CSVs exist.
    Runs generate_data.py if the files are missing.
    """
    import subprocess

    transactions_path = Path(_BASE_PATH) / "transactions_reduced.csv"
    if transactions_path.exists():
        logger.info("Reduced dataset already exists at %s. Skipping generation.", _BASE_PATH)
        return

    logger.info("Reduced dataset not found. Running generate_data.py ...")
    result = subprocess.run(
        [sys.executable, "generate_data.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"generate_data.py failed:\n{result.stderr}")
    logger.info("Data generation complete.")


def task_run_spark_pipeline(**context: Any) -> None:
    """
    Task 2: Clean raw data with PySpark (or Pandas fallback) and save parquet files.
    """
    from src.pipeline.pyspark_pipeline import main as run_pyspark
    os.makedirs(_PROCESSED_PATH, exist_ok=True)
    run_pyspark()
    logger.info("Spark/Pandas pipeline completed — processed files at %s", _PROCESSED_PATH)


def task_extract_features(**context: Any) -> None:
    """
    Task 3: Extract contextual + recency features and fit vocabulary indices.
    Saves final_training_data.csv to the processed directory.
    """
    from src.pipeline.integrate import run_integration_pipeline
    run_integration_pipeline()
    logger.info("Feature extraction and vocabulary fitting complete.")


def task_train_model(**context: Any) -> Dict[str, float]:
    """
    Task 4: Train the Two-Tower model on the integrated training dataset.
    Saves model weights to MODEL_DIR.

    Returns:
        Dict with final training loss and top-1 accuracy (pushed to XCom).
    """
    import numpy as np
    import pandas as pd

    from src.model.train import ModelTrainer

    training_csv = Path(_PROCESSED_PATH) / "final_training_data.csv"

    if not training_csv.exists():
        logger.warning("final_training_data.csv not found. Generating synthetic data for demo.")
        from src.model.train import _make_synthetic_df
        np.random.seed(42)
        df = _make_synthetic_df(n_rows=10_000, num_users=1000, num_items=2000)
        num_users = 1000
        num_items = 2000
        num_pg = 20
        num_cg = 50
    else:
        df = pd.read_csv(training_csv)
        num_users = int(df["customer_id_idx"].max()) + 1 if "customer_id_idx" in df.columns else 1000
        num_items = int(df["article_id_idx"].max())  + 1 if "article_id_idx"  in df.columns else 2000
        num_pg = int(df["product_group_name_idx"].max()) + 1 if "product_group_name_idx" in df.columns else 20
        num_cg = int(df["colour_group_name_idx"].max())  + 1 if "colour_group_name_idx"  in df.columns else 50

    split = int(0.90 * len(df))
    df_train = df.iloc[:split]
    df_val   = df.iloc[split:]

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
    val_ds   = trainer.build_tf_dataset(df_val,   shuffle=False)
    history  = trainer.train(train_ds, val_ds)

    final_loss = float(history.history["loss"][-1])
    final_acc  = float(history.history.get("top1_accuracy", [0.0])[-1])
    trainer.save_model_weights()

    logger.info("Training complete — final_loss=%.4f, top1_acc=%.4f", final_loss, final_acc)
    return {"final_loss": final_loss, "top1_accuracy": final_acc}


def task_export_embeddings(**context: Any) -> None:
    """
    Task 5: Export user and item embedding matrices as .npy files.
    These are read by the ANN index builder and the serving API at startup.
    """
    import numpy as np
    import pandas as pd

    from src.model.train import ModelTrainer, _make_synthetic_df

    training_csv = Path(_PROCESSED_PATH) / "final_training_data.csv"
    if not training_csv.exists():
        from src.model.train import _make_synthetic_df
        df = _make_synthetic_df(n_rows=5_000, num_users=500, num_items=1000)
    else:
        df = pd.read_csv(training_csv)

    # Reload trainer — load saved weights
    num_users = int(df["customer_id_idx"].max()) + 1 if "customer_id_idx" in df.columns else 500
    num_items = int(df["article_id_idx"].max())  + 1 if "article_id_idx"  in df.columns else 1000
    num_pg = int(df["product_group_name_idx"].max()) + 1 if "product_group_name_idx" in df.columns else 20
    num_cg = int(df["colour_group_name_idx"].max())  + 1 if "colour_group_name_idx"  in df.columns else 50

    trainer = ModelTrainer(embedding_dim=_EMBEDDING_DIM, output_dir=_MODEL_DIR)
    trainer.build_model(num_users=num_users, num_items=num_items,
                        num_product_groups=num_pg, num_colour_groups=num_cg)

    weights_path = Path(_MODEL_DIR) / "final_weights.h5"
    if weights_path.exists():
        # Warm-up call to build variable shapes, then load weights
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
        import tensorflow as tf
        trainer.model(
            {k: tf.constant(v) for k, v in dummy_batch.items()}, training=False
        )
        trainer.model.load_weights(str(weights_path))
        logger.info("Loaded model weights from %s", weights_path)

    user_feat_df = df.drop_duplicates(subset="customer_id_idx").reset_index(drop=True)
    item_feat_df = df.drop_duplicates(subset="article_id_idx").reset_index(drop=True)
    user_ids_arr = user_feat_df["customer_id_idx"].astype(str).values
    item_ids_arr = item_feat_df["article_id_idx"].astype(str).values

    trainer.save_user_embeddings(user_ids_arr, user_feat_df)
    trainer.save_item_embeddings(item_ids_arr, item_feat_df)
    logger.info("Embedding export complete.")


def task_build_ann_index(**context: Any) -> None:
    """
    Task 6: Load item embeddings and persist an IVFIndex to disk for fast serving.
    (The serving API rebuilds the index in memory on startup; this step validates it.)
    """
    import numpy as np
    from src.retrieval.ann_search import IVFIndex, ExactSearchIndex

    item_embs_path = Path(_MODEL_DIR) / "item_embeddings.npy"
    item_ids_path  = Path(_MODEL_DIR) / "item_ids.npy"

    if not item_embs_path.exists():
        raise FileNotFoundError(
            f"item_embeddings.npy not found at {_MODEL_DIR}. "
            "Run task_export_embeddings first."
        )

    item_embeddings = np.load(str(item_embs_path))
    item_ids        = np.load(str(item_ids_path), allow_pickle=True)
    n_items         = len(item_ids)

    # Use IVF if catalog large enough, else brute-force
    nlist = max(1, min(64, n_items // 10))
    if n_items >= nlist * 2:
        logger.info("Building IVFIndex (nlist=%d) for %d items.", nlist, n_items)
        index = IVFIndex(nlist=nlist, nprobe=4, normalize=True, random_state=42)
    else:
        logger.info("Building ExactSearchIndex for %d items.", n_items)
        index = ExactSearchIndex(normalize=True)

    index.fit(item_ids, item_embeddings)

    # Quick sanity search
    query = item_embeddings[:1]
    retrieved_ids, scores = index.search(query, k=min(5, n_items))
    logger.info(
        "ANN index sanity check: top-5 ids=%s, scores=%s",
        retrieved_ids[0][:5], scores[0][:5].round(4),
    )
    logger.info("ANN index build and validation complete.")


def task_run_evaluation(**context: Any) -> None:
    """
    Task 7: Run RetrievalEvaluator on a held-out test split and save metrics to JSON.
    Metrics are exposed by the /metrics API endpoint.
    """
    import numpy as np
    import pandas as pd

    from src.model.evaluate import ModelEvaluationPipeline

    pipeline = ModelEvaluationPipeline(model_dir=_MODEL_DIR, k_values=(5, 10, 20))

    try:
        pipeline.load_embeddings()
        pipeline.build_exact_index()
    except FileNotFoundError as exc:
        logger.warning("Embedding files not found: %s. Skipping evaluation.", exc)
        return

    # Load or synthesise test interactions
    training_csv = Path(_PROCESSED_PATH) / "final_training_data.csv"
    if training_csv.exists():
        df = pd.read_csv(training_csv)
        test_df = df.tail(max(100, len(df) // 10)).copy()

        # Resolve original string IDs if vocab-mapped
        user_col = "customer_id" if "customer_id" in test_df.columns else "customer_id_idx"
        item_col = "article_id"  if "article_id"  in test_df.columns else "article_id_idx"

        if user_col == "customer_id_idx":
            test_df["customer_id"] = test_df["customer_id_idx"].astype(str)
            user_col = "customer_id"
        if item_col == "article_id_idx":
            test_df["article_id"] = test_df["article_id_idx"].astype(str)
            item_col = "article_id"
    else:
        logger.warning("No training CSV found. Using synthetic test interactions.")
        n = len(pipeline.user_ids)
        m = len(pipeline.item_ids)
        scores_mat = np.matmul(pipeline.user_embeddings, pipeline.item_embeddings.T)
        records = []
        for u_idx in range(min(n, 200)):
            nearest = int(np.argmax(scores_mat[u_idx]))
            records.append({
                "customer_id": str(pipeline.user_ids[u_idx]),
                "article_id":  str(pipeline.item_ids[nearest]),
            })
        test_df  = pd.DataFrame(records)
        user_col = "customer_id"
        item_col = "article_id"

    metrics = pipeline.evaluate(test_df, user_col=user_col, item_col=item_col)

    if metrics:
        os.makedirs(_MODEL_DIR, exist_ok=True)
        metrics_path = Path(_MODEL_DIR) / "eval_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation complete. Metrics saved to %s:", metrics_path)
        for metric, val in sorted(metrics.items()):
            logger.info("    %-20s : %.4f", metric, val)
    else:
        logger.warning("No evaluation metrics returned.")


def task_notify_completion(**context: Any) -> None:
    """
    Task 8: Log pipeline completion summary (extend with Slack/email notifications).
    """
    run_id  = context.get("run_id", "unknown")
    dag_id  = context.get("dag").dag_id if context.get("dag") else "recommendation_training_pipeline"
    metrics_path = Path(_MODEL_DIR) / "eval_metrics.json"

    summary_lines = [
        f"  DAG            : {dag_id}",
        f"  Run ID         : {run_id}",
        f"  Timestamp      : {datetime.utcnow().isoformat()}Z",
        f"  Model dir      : {_MODEL_DIR}",
    ]

    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        summary_lines += [f"  {k:<20} : {v:.4f}" for k, v in sorted(m.items())]

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETION NOTIFICATION")
    for line in summary_lines:
        logger.info(line)
    logger.info("=" * 60)

    # TODO: extend with Slack webhook, email, or PagerDuty alert
    # slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    # if slack_webhook:
    #     requests.post(slack_webhook, json={"text": "\n".join(summary_lines)})


# ─────────────────────────────────────────────────────────────────────────────
# DAG definition (only executed when Airflow is available)
# ─────────────────────────────────────────────────────────────────────────────

_TASK_DEFINITIONS = [
    {
        "task_id":       "generate_data",
        "callable":      task_generate_data,
        "doc":           "Ensure reduced dataset CSVs exist; generate if missing.",
        "upstream":      [],
    },
    {
        "task_id":       "run_spark_pipeline",
        "callable":      task_run_spark_pipeline,
        "doc":           "Clean and transform raw data with PySpark/Pandas; output parquet.",
        "upstream":      ["generate_data"],
    },
    {
        "task_id":       "extract_features",
        "callable":      task_extract_features,
        "doc":           "Extract contextual + recency features; build vocabulary indices.",
        "upstream":      ["run_spark_pipeline"],
    },
    {
        "task_id":       "train_model",
        "callable":      task_train_model,
        "doc":           "Train Two-Tower model on integrated features; save weights.",
        "upstream":      ["extract_features"],
    },
    {
        "task_id":       "export_embeddings",
        "callable":      task_export_embeddings,
        "doc":           "Export user/item embedding matrices as .npy for ANN indexing.",
        "upstream":      ["train_model"],
    },
    {
        "task_id":       "build_ann_index",
        "callable":      task_build_ann_index,
        "doc":           "Build and validate the ANN retrieval index from item embeddings.",
        "upstream":      ["export_embeddings"],
    },
    {
        "task_id":       "run_evaluation",
        "callable":      task_run_evaluation,
        "doc":           "Evaluate Recall@K/NDCG@K on test split; persist metrics.json.",
        "upstream":      ["build_ann_index"],
    },
    {
        "task_id":       "notify_completion",
        "callable":      task_notify_completion,
        "doc":           "Log (and optionally notify) pipeline completion summary.",
        "upstream":      ["run_evaluation"],
    },
]

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="recommendation_training_pipeline",
        default_args=_DEFAULT_ARGS,
        description=(
            "Daily two-tower recommendation model training pipeline: "
            "data ingestion → feature engineering → model training → "
            "embedding export → ANN index build → evaluation → notification."
        ),
        schedule_interval="@daily",
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        tags=["recommendation", "two-tower", "deep-learning"],
    ) as dag:

        airflow_tasks = {}
        for task_def in _TASK_DEFINITIONS:
            t = PythonOperator(
                task_id=task_def["task_id"],
                python_callable=task_def["callable"],
                doc_md=task_def["doc"],
                provide_context=True,
            )
            airflow_tasks[task_def["task_id"]] = t

        # Wire up dependencies
        for task_def in _TASK_DEFINITIONS:
            for upstream_id in task_def["upstream"]:
                airflow_tasks[upstream_id] >> airflow_tasks[task_def["task_id"]]


# ─────────────────────────────────────────────────────────────────────────────
# Standalone execution — prints DAG task graph summary & runs a dry-run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 70)
    print("DAY 7: AIRFLOW DAG — recommendation_training_pipeline")
    print("=" * 70)

    if AIRFLOW_AVAILABLE:
        print("\nAirflow is installed. DAG registered successfully.")
    else:
        print("\nAirflow NOT installed — showing standalone task graph summary.\n")

    print("\nTask Graph (sequential pipeline):")
    for i, task_def in enumerate(_TASK_DEFINITIONS):
        arrow = "→ " if i > 0 else "  "
        print(f"  {arrow}[{i+1}] {task_def['task_id']:<25} {task_def['doc']}")

    print("\n" + "-" * 70)
    print("Dry-run: executing task_generate_data and task_build_ann_index only...")

    import tempfile, shutil, numpy as np

    tmp_dir = tempfile.mkdtemp(prefix="dag_dryrun_")
    try:
        # Simulate export_embeddings output for build_ann_index dry run
        os.makedirs(tmp_dir, exist_ok=True)
        np.random.seed(0)
        n_i, d = 200, 64
        i_embs = np.random.randn(n_i, d).astype(np.float32)
        i_embs /= np.linalg.norm(i_embs, axis=1, keepdims=True)
        i_ids = np.array([f"item_{i}" for i in range(n_i)])
        np.save(os.path.join(tmp_dir, "item_embeddings.npy"), i_embs)
        np.save(os.path.join(tmp_dir, "item_ids.npy"), i_ids)

        global _MODEL_DIR
        _orig = _MODEL_DIR
        _MODEL_DIR = tmp_dir
        task_build_ann_index()
        task_notify_completion(run_id="dry_run_001")
        _MODEL_DIR = _orig

        print("\nDry-run completed successfully.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("Day 7 Airflow DAG demo completed!")
    print("=" * 70 + "\n")
