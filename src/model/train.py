"""
Day 6: Model Trainer — Training Loop, Dataset Builder & Embedding Export.

ModelTrainer orchestrates the full training lifecycle of the TwoTowerModel:
    1. Build a tf.data.Dataset from either the integrated CSV or synthetic data
    2. Compile and fit the model (Adam, learning-rate schedule, early stopping)
    3. Export user and item embedding matrices as .npy files for offline ANN indexing

The exported .npy files plug directly into Day 4's ExactSearchIndex / IVFIndex for
low-latency retrieval at serving time.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.query_tower import QueryTower
from src.model.candidate_tower import CandidateTower
from src.model.two_tower_model import TwoTowerModel

logger = logging.getLogger(__name__)

# ─── Columns needed from the integrated training dataset ──────────────────────
_USER_FEATURE_COLS = [
    "customer_id_idx",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "days_since_last_purchase",
    "purchase_sequence",
]
_ITEM_FEATURE_COLS = [
    "article_id_idx",
    "product_group_name_idx",
    "colour_group_name_idx",
    "popularity_over_time",
]
_ALL_FEATURE_COLS = _USER_FEATURE_COLS + _ITEM_FEATURE_COLS


class ModelTrainer:
    """
    Training coordinator for the Two-Tower Recommendation Model.

    Handles dataset preparation, model construction, training loop, and
    artefact serialisation (model weights + embedding matrices).
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        batch_size: int = 256,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        temperature: float = 0.05,
        patience: int = 3,
        output_dir: str = "data/processed/model",
    ):
        """
        Initialise ModelTrainer.

        Args:
            embedding_dim: Shared output embedding size for both towers.
            batch_size: Training batch size (determines in-batch negatives count).
            epochs: Maximum training epochs.
            learning_rate: Initial Adam learning rate.
            temperature: Softmax temperature for in-batch loss.
            patience: Early-stopping patience (epochs without improvement).
            output_dir: Directory to save weights, embeddings, and config.
        """
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.patience = patience
        self.output_dir = Path(output_dir)

        self.model: Optional[TwoTowerModel] = None
        self.vocab_sizes: Dict[str, int] = {}
        self.user_ids: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset construction
    # ─────────────────────────────────────────────────────────────────────────

    def _df_to_feature_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convert a Pandas DataFrame to a dictionary of NumPy arrays matching the
        expected feature keys for the TwoTowerModel.

        Args:
            df: DataFrame with columns defined in _ALL_FEATURE_COLS.

        Returns:
            Dict mapping feature name → np.ndarray.
        """
        feat = {}
        int_cols = {"customer_id_idx", "article_id_idx",
                    "product_group_name_idx", "colour_group_name_idx"}
        for col in _ALL_FEATURE_COLS:
            if col in df.columns:
                if col in int_cols:
                    feat[col] = df[col].fillna(0).astype(np.int32).values
                else:
                    feat[col] = df[col].fillna(0.0).astype(np.float32).values
            else:
                # Provide sensible default if column missing (e.g. synthetic data)
                if col in int_cols:
                    feat[col] = np.zeros(len(df), dtype=np.int32)
                else:
                    feat[col] = np.zeros(len(df), dtype=np.float32)
        return feat

    def build_tf_dataset(
        self,
        df: pd.DataFrame,
        shuffle: bool = True,
        drop_remainder: bool = True,
    ) -> tf.data.Dataset:
        """
        Build a batched tf.data.Dataset from a Pandas DataFrame.

        Args:
            df: Interaction DataFrame with features.
            shuffle: Whether to shuffle the dataset each epoch.
            drop_remainder: Drop the last partial batch (keeps in-batch negatives uniform).

        Returns:
            tf.data.Dataset yielding feature dictionaries.
        """
        feat_dict = self._df_to_feature_dict(df)
        dataset = tf.data.Dataset.from_tensor_slices(feat_dict)

        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=min(len(df), 10_000), reshuffle_each_iteration=True
            )

        dataset = dataset.batch(self.batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # ─────────────────────────────────────────────────────────────────────────
    # Model construction
    # ─────────────────────────────────────────────────────────────────────────

    def build_model(
        self,
        num_users: int,
        num_items: int,
        num_product_groups: int = 20,
        num_colour_groups: int = 50,
    ) -> TwoTowerModel:
        """
        Instantiate and compile TwoTowerModel.

        Args:
            num_users: Vocabulary size for customer IDs.
            num_items: Vocabulary size for article IDs.
            num_product_groups: Vocabulary size for product_group_name.
            num_colour_groups: Vocabulary size for colour_group_name.

        Returns:
            Compiled TwoTowerModel.
        """
        self.vocab_sizes = {
            "num_users": num_users,
            "num_items": num_items,
            "num_product_groups": num_product_groups,
            "num_colour_groups": num_colour_groups,
        }

        query_tower = QueryTower(
            num_users=num_users,
            embedding_dim=self.embedding_dim,
        )
        candidate_tower = CandidateTower(
            num_items=num_items,
            num_product_groups=num_product_groups,
            num_colour_groups=num_colour_groups,
            embedding_dim=self.embedding_dim,
        )
        self.model = TwoTowerModel(
            query_tower=query_tower,
            candidate_tower=candidate_tower,
            temperature=self.temperature,
        )

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1000,
            alpha=0.01,
        )
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule))

        logger.info(
            "TwoTowerModel built — users=%d, items=%d, emb_dim=%d",
            num_users, num_items, self.embedding_dim,
        )
        return self.model

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
    ) -> tf.keras.callbacks.History:
        """
        Fit the model with early stopping and learning-rate reduction.

        Args:
            train_dataset: Training tf.data.Dataset.
            val_dataset: Optional validation tf.data.Dataset.

        Returns:
            Keras History object.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before train().")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_dataset is not None else "loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.output_dir / "best_weights.h5"),
                monitor="val_loss" if val_dataset is not None else "loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
        ]

        logger.info("Starting training — epochs=%d, batch_size=%d", self.epochs, self.batch_size)
        t0 = time.perf_counter()

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        elapsed = time.perf_counter() - t0
        logger.info("Training completed in %.1f seconds.", elapsed)
        return history

    # ─────────────────────────────────────────────────────────────────────────
    # Embedding export
    # ─────────────────────────────────────────────────────────────────────────

    def save_user_embeddings(
        self,
        user_id_array: np.ndarray,
        user_feature_df: pd.DataFrame,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export all user embedding vectors by running the QueryTower in inference mode.

        Args:
            user_id_array: Array of unique user_id strings (N,).
            user_feature_df: DataFrame with one row per unique user, containing all
                             user-side feature columns. Index must align with user_id_array.
            batch_size: Embedding inference batch size.

        Returns:
            Tuple of (user_ids, user_embeddings) where embeddings has shape (N, D).
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        feat_dict = self._df_to_feature_dict(user_feature_df)
        n = len(user_id_array)
        all_embs = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {k: tf.constant(v[start:end]) for k, v in feat_dict.items()
                     if k in _USER_FEATURE_COLS}
            emb = self.model.get_user_embedding(batch, training=False).numpy()
            all_embs.append(emb)

        user_embeddings = np.vstack(all_embs)
        self.user_ids = user_id_array

        np.save(str(self.output_dir / "user_ids.npy"), user_id_array)
        np.save(str(self.output_dir / "user_embeddings.npy"), user_embeddings)
        logger.info(
            "Saved user embeddings: shape=%s → %s",
            user_embeddings.shape, self.output_dir,
        )
        return user_id_array, user_embeddings

    def save_item_embeddings(
        self,
        item_id_array: np.ndarray,
        item_feature_df: pd.DataFrame,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Export all item embedding vectors by running the CandidateTower in inference mode.

        Args:
            item_id_array: Array of unique article_id strings (M,).
            item_feature_df: DataFrame with one row per unique item.
            batch_size: Embedding inference batch size.

        Returns:
            Tuple of (item_ids, item_embeddings) where embeddings has shape (M, D).
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        feat_dict = self._df_to_feature_dict(item_feature_df)
        m = len(item_id_array)
        all_embs = []

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch = {k: tf.constant(v[start:end]) for k, v in feat_dict.items()
                     if k in _ITEM_FEATURE_COLS}
            emb = self.model.get_item_embedding(batch, training=False).numpy()
            all_embs.append(emb)

        item_embeddings = np.vstack(all_embs)
        self.item_ids = item_id_array

        np.save(str(self.output_dir / "item_ids.npy"), item_id_array)
        np.save(str(self.output_dir / "item_embeddings.npy"), item_embeddings)
        logger.info(
            "Saved item embeddings: shape=%s → %s",
            item_embeddings.shape, self.output_dir,
        )
        return item_id_array, item_embeddings

    def save_model_weights(self) -> None:
        """Save final model weights to output_dir/final_weights.h5."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = str(self.output_dir / "final_weights.h5")
        self.model.save_weights(path)
        logger.info("Final model weights saved → %s", path)

    @classmethod
    def load_embeddings(cls, model_dir: str = "data/processed/model") -> Dict[str, np.ndarray]:
        """
        Load previously exported embedding arrays from disk.

        Args:
            model_dir: Directory containing the .npy files.

        Returns:
            Dictionary with keys 'user_ids', 'user_embeddings', 'item_ids', 'item_embeddings'.
        """
        dirpath = Path(model_dir)
        artefacts = {}
        for key in ("user_ids", "user_embeddings", "item_ids", "item_embeddings"):
            fpath = dirpath / f"{key}.npy"
            if fpath.exists():
                artefacts[key] = np.load(str(fpath), allow_pickle=True)
                logger.info("Loaded %s: shape=%s", key, artefacts[key].shape)
            else:
                logger.warning("Embedding file not found: %s", fpath)
        return artefacts


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo — trains on synthetic data and exports embeddings
# ─────────────────────────────────────────────────────────────────────────────
def _make_synthetic_df(n_rows: int, num_users: int, num_items: int) -> pd.DataFrame:
    """Generate synthetic training interactions DataFrame."""
    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, n_rows)
    return pd.DataFrame({
        "customer_id_idx":            rng.integers(1, num_users + 1, n_rows).astype(np.int32),
        "month_sin":                  np.sin(angles).astype(np.float32),
        "month_cos":                  np.cos(angles).astype(np.float32),
        "day_of_week_sin":            rng.uniform(-1, 1, n_rows).astype(np.float32),
        "day_of_week_cos":            rng.uniform(-1, 1, n_rows).astype(np.float32),
        "is_weekend":                 rng.integers(0, 2, n_rows).astype(np.float32),
        "days_since_last_purchase":   rng.uniform(0, 60, n_rows).astype(np.float32),
        "purchase_sequence":          rng.integers(1, 30, n_rows).astype(np.float32),
        "article_id_idx":             rng.integers(1, num_items + 1, n_rows).astype(np.int32),
        "product_group_name_idx":     rng.integers(1, 21, n_rows).astype(np.int32),
        "colour_group_name_idx":      rng.integers(1, 51, n_rows).astype(np.int32),
        "popularity_over_time":       rng.uniform(1, 200, n_rows).astype(np.float32),
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 70)
    print("DAY 6: TWO-TOWER MODEL TRAINING PIPELINE DEMO")
    print("=" * 70)

    tf.random.set_seed(42)
    np.random.seed(42)

    NUM_USERS  = 300
    NUM_ITEMS  = 500
    N_ROWS     = 5_000

    print(f"\n[1] Generating synthetic dataset ({N_ROWS} interactions)...")
    df_all = _make_synthetic_df(N_ROWS, NUM_USERS, NUM_ITEMS)
    split = int(0.85 * N_ROWS)
    df_train = df_all.iloc[:split]
    df_val   = df_all.iloc[split:]
    print(f"    Train: {len(df_train)}, Val: {len(df_val)}")

    print("\n[2] Initialising ModelTrainer...")
    trainer = ModelTrainer(
        embedding_dim=64,
        batch_size=128,
        epochs=5,
        learning_rate=1e-3,
        temperature=0.05,
        patience=2,
        output_dir="data/processed/model",
    )

    print("\n[3] Building TwoTowerModel...")
    trainer.build_model(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        num_product_groups=20,
        num_colour_groups=50,
    )

    print("\n[4] Building tf.data pipelines...")
    train_ds = trainer.build_tf_dataset(df_train, shuffle=True)
    val_ds   = trainer.build_tf_dataset(df_val, shuffle=False)

    print("\n[5] Training model...")
    history = trainer.train(train_ds, val_ds)

    final_loss = history.history["loss"][-1]
    final_acc  = history.history["top1_accuracy"][-1]
    print(f"\n    Final Train Loss      : {final_loss:.4f}")
    print(f"    Final Top-1 Accuracy  : {final_acc:.4f}")

    print("\n[6] Exporting user & item embeddings...")
    # Unique user / item feature snapshots (last occurrence per user/item)
    user_feat_df = df_all.drop_duplicates(subset="customer_id_idx") \
                         .sort_values("customer_id_idx").reset_index(drop=True)
    item_feat_df = df_all.drop_duplicates(subset="article_id_idx") \
                         .sort_values("article_id_idx").reset_index(drop=True)

    user_ids_arr = user_feat_df["customer_id_idx"].astype(str).values
    item_ids_arr = item_feat_df["article_id_idx"].astype(str).values

    u_ids, u_embs = trainer.save_user_embeddings(user_ids_arr, user_feat_df)
    i_ids, i_embs = trainer.save_item_embeddings(item_ids_arr, item_feat_df)

    print(f"\n    User embeddings : {u_embs.shape}")
    print(f"    Item embeddings : {i_embs.shape}")

    trainer.save_model_weights()

    print("\n[7] Verifying embedding norms (should all ≈ 1.0)...")
    u_norms = np.linalg.norm(u_embs, axis=1)
    i_norms = np.linalg.norm(i_embs, axis=1)
    print(f"    User norm mean/std : {u_norms.mean():.4f} / {u_norms.std():.6f}")
    print(f"    Item norm mean/std : {i_norms.mean():.4f} / {i_norms.std():.6f}")

    print("\n" + "=" * 70)
    print("Day 6 Training Pipeline Demo Completed Successfully!")
    print("=" * 70 + "\n")
