"""
Day 6: Query Tower — User + Context Encoder for Two-Tower Neural Recommendation Model.

The QueryTower maps a user's identity features and contextual signals (season, recency,
purchase frequency, weekend/channel context) into a dense L2-normalised embedding that
can be compared with candidate embeddings via dot-product similarity.

Input features (from final_training_data.csv / synthetic batch dict):
    - customer_id_idx    : int   — vocabulary index for user identity embedding
    - month_sin          : float — cyclical month feature (sin)
    - month_cos          : float — cyclical month feature (cos)
    - day_of_week_sin    : float — cyclical day-of-week feature (sin)
    - day_of_week_cos    : float — cyclical day-of-week feature (cos)
    - is_weekend         : float — binary flag (0 or 1)
    - days_since_last_purchase : float — recency signal
    - purchase_sequence  : float — cumulative purchase count (frequency proxy)

Output:
    Tensor of shape (batch_size, embedding_dim), L2-normalised.

Note: TensorFlow is required. If not installed, import will raise ImportError.
"""

import os
import sys
import logging
from typing import Dict, Optional

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info/warning logs

try:
    import tensorflow as tf
except ImportError as _tf_err:
    raise ImportError(
        "TensorFlow is required for the Two-Tower model. "
        "Install it with: pip install tensorflow>=2.15"
    ) from _tf_err

# Append workspace root for standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logger = logging.getLogger(__name__)

# Contextual feature dimension: month_sin, month_cos, dow_sin, dow_cos, is_weekend,
# days_since_last_purchase, purchase_sequence  →  7 continuous features
_NUM_CONTEXT_FEATURES = 7


class QueryTower(tf.keras.Model):
    """
    User + Context Encoder Tower for the Two-Tower Retrieval Model.

    Architecture:
        user_id → Embedding(num_users, user_emb_dim)
            ↓
        context_features → Dense(context_proj_dim, relu)
            ↓
        concat([user_emb, context_proj])
            ↓
        Dense(hidden_dim, relu) → BatchNorm
            ↓
        Dense(hidden_dim // 2, relu)
            ↓
        Dense(embedding_dim)
            ↓
        L2-normalise → output embedding
    """

    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        user_emb_dim: int = 32,
        context_proj_dim: int = 16,
        hidden_dim: int = 128,
        dropout_rate: float = 0.10,
        name: str = "query_tower",
        **kwargs,
    ):
        """
        Initialise QueryTower.

        Args:
            num_users: Vocabulary size for user IDs (+ 1 for UNK at index 0).
            embedding_dim: Output embedding dimensionality (shared with CandidateTower).
            user_emb_dim: Dimensionality of the user identity embedding table.
            context_proj_dim: Hidden size for the context projection branch.
            hidden_dim: Hidden size of the main MLP body.
            dropout_rate: Dropout probability for regularisation.
            name: Layer name.
        """
        super().__init__(name=name, **kwargs)

        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.user_emb_dim = user_emb_dim
        self.context_proj_dim = context_proj_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # ── User identity branch ──────────────────────────────────────────────
        self.user_embedding = tf.keras.layers.Embedding(
            input_dim=num_users + 1,   # +1 for UNK (index 0)
            output_dim=user_emb_dim,
            embeddings_initializer="glorot_uniform",
            embeddings_regularizer=tf.keras.regularizers.L2(1e-5),
            name="user_embedding",
        )

        # ── Context features branch ───────────────────────────────────────────
        self.context_norm = tf.keras.layers.LayerNormalization(name="ctx_norm")
        self.context_proj = tf.keras.layers.Dense(
            context_proj_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="ctx_proj",
        )

        # ── Main MLP body ─────────────────────────────────────────────────────
        self.dense1 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="dense1",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")

        self.dense2 = tf.keras.layers.Dense(
            hidden_dim // 2,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="dense2",
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

        # ── Output projection ─────────────────────────────────────────────────
        self.output_proj = tf.keras.layers.Dense(
            embedding_dim,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="output_proj",
        )

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """
        Forward pass.

        Args:
            inputs: Dictionary containing:
                - 'customer_id_idx'         : int32 Tensor (batch,)
                - 'month_sin'               : float32 Tensor (batch,)
                - 'month_cos'               : float32 Tensor (batch,)
                - 'day_of_week_sin'         : float32 Tensor (batch,)
                - 'day_of_week_cos'         : float32 Tensor (batch,)
                - 'is_weekend'              : float32 Tensor (batch,)
                - 'days_since_last_purchase': float32 Tensor (batch,)
                - 'purchase_sequence'       : float32 Tensor (batch,)
            training: Boolean flag for dropout / batch-norm behaviour.

        Returns:
            tf.Tensor of shape (batch, embedding_dim), L2-normalised.
        """
        # ── User embedding ────────────────────────────────────────────────────
        user_idx = tf.cast(inputs["customer_id_idx"], tf.int32)
        user_emb = self.user_embedding(user_idx)                 # (B, user_emb_dim)

        # ── Context feature assembly ──────────────────────────────────────────
        ctx = tf.stack(
            [
                tf.cast(inputs["month_sin"], tf.float32),
                tf.cast(inputs["month_cos"], tf.float32),
                tf.cast(inputs["day_of_week_sin"], tf.float32),
                tf.cast(inputs["day_of_week_cos"], tf.float32),
                tf.cast(inputs["is_weekend"], tf.float32),
                tf.cast(inputs["days_since_last_purchase"], tf.float32),
                tf.cast(inputs["purchase_sequence"], tf.float32),
            ],
            axis=1,
        )                                                          # (B, 7)
        ctx = self.context_norm(ctx, training=training)
        ctx_proj = self.context_proj(ctx, training=training)      # (B, context_proj_dim)

        # ── Concatenate and pass through MLP ──────────────────────────────────
        x = tf.concat([user_emb, ctx_proj], axis=1)              # (B, user_emb_dim + context_proj_dim)

        x = self.dense1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.output_proj(x)                                   # (B, embedding_dim)

        # ── L2-normalise output ───────────────────────────────────────────────
        x = tf.math.l2_normalize(x, axis=-1)
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            {
                "num_users": self.num_users,
                "embedding_dim": self.embedding_dim,
                "user_emb_dim": self.user_emb_dim,
                "context_proj_dim": self.context_proj_dim,
                "hidden_dim": self.hidden_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n" + "=" * 70)
    print("DAY 6: QUERY TOWER ARCHITECTURE DEMO")
    print("=" * 70)

    tf.random.set_seed(42)
    batch_size = 8
    num_users = 500

    tower = QueryTower(num_users=num_users, embedding_dim=64)

    dummy_inputs = {
        "customer_id_idx":           tf.random.uniform((batch_size,), 0, num_users, dtype=tf.int32),
        "month_sin":                 tf.random.normal((batch_size,)),
        "month_cos":                 tf.random.normal((batch_size,)),
        "day_of_week_sin":           tf.random.normal((batch_size,)),
        "day_of_week_cos":           tf.random.normal((batch_size,)),
        "is_weekend":                tf.random.uniform((batch_size,), 0, 2, dtype=tf.float32),
        "days_since_last_purchase":  tf.random.normal((batch_size,)),
        "purchase_sequence":         tf.random.uniform((batch_size,), 1, 50, dtype=tf.float32),
    }

    output = tower(dummy_inputs, training=False)

    print(f"\n  Input batch size   : {batch_size}")
    print(f"  Output shape       : {output.shape}   (expected: ({batch_size}, 64))")
    norms = tf.norm(output, axis=-1).numpy()
    print(f"  L2 norms (all ≈1) : {norms.round(4)}")

    tower.summary()
    print("\nQuery Tower demo completed successfully!")
