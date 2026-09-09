"""
Day 6: Two-Tower Neural Recommendation Model.

Combines QueryTower (user + context encoder) and CandidateTower (item encoder)
into a unified model trained with in-batch sampled softmax loss.

Training objective:
    For a batch of (query, positive_candidate) pairs, the positive pair is on the
    diagonal of the score matrix  S = Q · C^T  (shape: batch × batch).
    Softmax cross-entropy over each row (with temperature scaling) turns every other
    item in the batch into a sampled negative — an efficient, scalable approximation
    to full softmax over the entire catalog.

At inference:
    - `get_user_embedding(inputs)` → query embedding for a single user+context
    - `get_item_embedding(inputs)` → candidate embedding for a single item
    - Offline: export all item embeddings → build ANN index (ExactSearchIndex / IVFIndex)
    - Online: encode user query → nearest-neighbour search against the index
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.query_tower import QueryTower
from src.model.candidate_tower import CandidateTower

logger = logging.getLogger(__name__)

# Default log-temperature for in-batch sampled softmax
_DEFAULT_TEMPERATURE = 0.05


class TwoTowerModel(tf.keras.Model):
    """
    Dual-Encoder (Two-Tower) Retrieval Model with In-Batch Sampled Softmax Loss.

    The model learns embeddings such that:
        similarity(query(u, ctx), candidate(i)) > similarity(query(u, ctx), candidate(j))
    for all positive items i relative to in-batch negatives j.
    """

    def __init__(
        self,
        query_tower: QueryTower,
        candidate_tower: CandidateTower,
        temperature: float = _DEFAULT_TEMPERATURE,
        name: str = "two_tower_model",
        **kwargs,
    ):
        """
        Initialise TwoTowerModel.

        Args:
            query_tower: Pre-instantiated QueryTower encoder.
            candidate_tower: Pre-instantiated CandidateTower encoder.
            temperature: Softmax temperature τ; lower values → sharper distributions.
            name: Model name.
        """
        super().__init__(name=name, **kwargs)
        self.query_tower = query_tower
        self.candidate_tower = candidate_tower
        self.temperature = temperature

        # Track training metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.top1_accuracy_tracker = tf.keras.metrics.Mean(name="top1_accuracy")

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────
# to compute both query and candidate embeddings using arguments such as inputs and trainings
    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute both query and candidate embeddings for a batch.

        Args:
            inputs: Feature dictionary containing both user-side and item-side keys.
            training: Training mode flag (affects dropout / batch-norm).

        Returns:
            Tuple of (query_embeddings, candidate_embeddings), each shape (B, D).
        """
        query_emb = self.query_tower(inputs, training=training)
        candidate_emb = self.candidate_tower(inputs, training=training)
        return query_emb, candidate_emb

    # ─────────────────────────────────────────────────────────────────────────
    # Loss computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss_from_embeddings(
        self,
        query_emb: tf.Tensor,
        candidate_emb: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute in-batch sampled softmax loss and top-1 accuracy.

        Score matrix:  S[i, j] = dot(q_i, c_j) / τ
        Labels:        identity matrix  (diagonal = positive pairs)
        Loss:          mean cross-entropy over rows of S

        Args:
            query_emb: (B, D) normalised query embeddings.
            candidate_emb: (B, D) normalised candidate embeddings.

        Returns:
            Tuple of (loss_scalar, top1_accuracy_scalar).
        """
        # Score matrix: (B, B)
        scores = tf.matmul(query_emb, candidate_emb, transpose_b=True) / self.temperature

        # Diagonal labels (each query's positive is at position i in row i)
        batch_size = tf.shape(scores)[0]
        labels = tf.eye(batch_size, dtype=tf.float32)

        # Row-wise softmax cross-entropy
        loss = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                y_true=labels,
                y_pred=scores,
                from_logits=True,
            )
        )

        # Top-1 accuracy: fraction of queries where argmax score is the positive
        top1_preds = tf.cast(tf.argmax(scores, axis=1), tf.int32)
        true_labels = tf.range(batch_size, dtype=tf.int32)
        top1_acc = tf.reduce_mean(
            tf.cast(tf.equal(top1_preds, true_labels), tf.float32)
        )

        return loss, top1_acc


    # Training & evaluation step overrides


    def train_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Single gradient-descent training step.

        Args:
            data: Feature dictionary (same as `call` inputs).

        Returns:
            Dictionary of metric names → scalar values.
        """
        with tf.GradientTape() as tape:
            query_emb, candidate_emb = self(data, training=True)
            loss, top1_acc = self.compute_loss_from_embeddings(query_emb, candidate_emb)

            # Include regularisation losses from sub-layers
            reg_loss = tf.add_n(self.losses) if self.losses else 0.0
            total_loss = loss + reg_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.top1_accuracy_tracker.update_state(top1_acc)

        return {
            "loss": self.loss_tracker.result(),
            "top1_accuracy": self.top1_accuracy_tracker.result(),
        }

    def test_step(self, data: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        Single validation step (no gradient computation).

        Args:
            data: Feature dictionary.

        Returns:
            Dictionary of metric names → scalar values.
        """
        query_emb, candidate_emb = self(data, training=False)
        loss, top1_acc = self.compute_loss_from_embeddings(query_emb, candidate_emb)
        reg_loss = tf.add_n(self.losses) if self.losses else 0.0
        total_loss = loss + reg_loss

        self.loss_tracker.update_state(total_loss)
        self.top1_accuracy_tracker.update_state(top1_acc)

        return {
            "loss": self.loss_tracker.result(),
            "top1_accuracy": self.top1_accuracy_tracker.result(),
        }

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [self.loss_tracker, self.top1_accuracy_tracker]

    # ─────────────────────────────────────────────────────────────────────────
    # Inference helpers
    # ─────────────────────────────────────────────────────────────────────────
# to get the user embeddings
    def get_user_embedding(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        """
        Compute query embedding for a user + context input dictionary.

        Args:
            inputs: Feature dict (user + context features).
            training: Training mode flag.

        Returns:
            tf.Tensor of shape (B, embedding_dim), L2-normalised.
        """
        return self.query_tower(inputs, training=training)
#to get the item and embed
    def get_item_embedding(
        self, inputs: Dict[str, tf.Tensor], training: bool = False
    ) -> tf.Tensor:
        """
        Compute candidate embedding for an item feature input dictionary.

        Args:
            inputs: Feature dict (item features).
            training: Training mode flag.

        Returns:
            tf.Tensor of shape (B, embedding_dim), L2-normalised.
        """
        return self.candidate_tower(inputs, training=training)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["temperature"] = self.temperature
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 70)
    print("DAY 6: TWO-TOWER MODEL — ARCHITECTURE & FORWARD PASS DEMO")
    print("=" * 70)

    tf.random.set_seed(42)
    np.random.seed(42)

    NUM_USERS  = 500
    NUM_ITEMS  = 1000
    BATCH_SIZE = 32

    # Instantiate towers
    query_tower = QueryTower(num_users=NUM_USERS, embedding_dim=64)
    candidate_tower = CandidateTower(
        num_items=NUM_ITEMS,
        num_product_groups=20,
        num_colour_groups=50,
        embedding_dim=64,
    )
    model = TwoTowerModel(query_tower, candidate_tower, temperature=0.05)

    # Synthetic batch
    dummy_batch = {
        # User / context features
        "customer_id_idx":           tf.random.uniform((BATCH_SIZE,), 0, NUM_USERS, dtype=tf.int32),
        "month_sin":                 tf.random.normal((BATCH_SIZE,)),
        "month_cos":                 tf.random.normal((BATCH_SIZE,)),
        "day_of_week_sin":           tf.random.normal((BATCH_SIZE,)),
        "day_of_week_cos":           tf.random.normal((BATCH_SIZE,)),
        "is_weekend":                tf.random.uniform((BATCH_SIZE,), 0, 2, dtype=tf.float32),
        "days_since_last_purchase":  tf.random.normal((BATCH_SIZE,)),
        "purchase_sequence":         tf.random.uniform((BATCH_SIZE,), 1, 50, dtype=tf.float32),
        # Item / candidate features
        "article_id_idx":            tf.random.uniform((BATCH_SIZE,), 0, NUM_ITEMS, dtype=tf.int32),
        "product_group_name_idx":    tf.random.uniform((BATCH_SIZE,), 0, 20, dtype=tf.int32),
        "colour_group_name_idx":     tf.random.uniform((BATCH_SIZE,), 0, 50, dtype=tf.int32),
        "popularity_over_time":      tf.random.uniform((BATCH_SIZE,), 1.0, 200.0, dtype=tf.float32),
    }

    # Forward pass
    q_emb, c_emb = model(dummy_batch, training=False)
    print(f"\n  Query embedding shape     : {q_emb.shape}")
    print(f"  Candidate embedding shape : {c_emb.shape}")

    # Compute loss
    loss_val, acc_val = model.compute_loss_from_embeddings(q_emb, c_emb)
    print(f"\n  In-batch softmax loss     : {loss_val.numpy():.4f}")
    print(f"  Top-1 accuracy (random)   : {acc_val.numpy():.4f}  "
          f"(expected ≈ 1/{BATCH_SIZE} = {1/BATCH_SIZE:.4f})")

    # Compile and single train step
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    metrics_out = model.train_step(dummy_batch)
    print(f"\n  After 1 train step — loss: {metrics_out['loss'].numpy():.4f}, "
          f"top1_acc: {metrics_out['top1_accuracy'].numpy():.4f}")

    print("\nTwo-Tower Model demo completed successfully!")
