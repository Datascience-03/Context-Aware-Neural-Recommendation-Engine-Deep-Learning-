
"""
Candidate Tower — Item Feature Encoder for Two-Tower Neural Recommendation Model.

The CandidateTower maps article features (identity, product group, colour, popularity)
into a dense L2-normalised embedding in the same space as the QueryTower output,
enabling efficient dot-product retrieval at inference time.

Input features (from final_training_data.csv / synthetic batch dict):
    - article_id_idx           : int   — vocabulary index for item identity embedding
    - product_group_name_idx   : int   — product group category embedding
    - colour_group_name_idx    : int   — colour group category embedding
    - popularity_over_time     : float — rolling 30-day sales count (popularity signal)

Output:
    Tensor of shape (batch_size, embedding_dim), L2-normalised.
"""

import os
import sys
import logging
from typing import Dict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logger = logging.getLogger(__name__)


class CandidateTower(tf.keras.Model):
    """
    Item Feature Encoder Tower for the Two-Tower Retrieval Model.

    Architecture:
        article_id → Embedding(num_items, item_emb_dim)
        product_group_idx → Embedding(num_product_groups, cat_emb_dim)
        colour_group_idx  → Embedding(num_colour_groups,  cat_emb_dim)
        popularity_over_time → Dense(cat_emb_dim, relu)
            ↓
        concat([item_emb, product_emb, colour_emb, pop_proj])
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
        num_items: int,
        num_product_groups: int = 20,
        num_colour_groups: int = 50,
        embedding_dim: int = 64,
        item_emb_dim: int = 32,
        cat_emb_dim: int = 8,
        hidden_dim: int = 128,
        dropout_rate: float = 0.10,
        name: str = "candidate_tower",
        **kwargs,
    ):
        """
        Initialise CandidateTower.

        Args:
            num_items: Vocabulary size for article IDs.
            num_product_groups: Vocabulary size for product group names.
            num_colour_groups: Vocabulary size for colour group names.
            embedding_dim: Output embedding dimensionality (shared with QueryTower).
            item_emb_dim: Dimensionality of the item identity embedding table.
            cat_emb_dim: Dimensionality of each side-feature categorical embedding.
            hidden_dim: Hidden size of the main MLP body.
            dropout_rate: Dropout probability for regularisation.
            name: Layer name.
        """
        super().__init__(name=name, **kwargs)

        self.num_items = num_items
        self.num_product_groups = num_product_groups
        self.num_colour_groups = num_colour_groups
        self.embedding_dim = embedding_dim
        self.item_emb_dim = item_emb_dim
        self.cat_emb_dim = cat_emb_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # ── Item identity embedding ───────────────────────────────────────────
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items + 1,
            output_dim=item_emb_dim,
            embeddings_initializer="glorot_uniform",
            embeddings_regularizer=tf.keras.regularizers.L2(1e-5),
            name="item_embedding",
        )

        # ── Product group embedding ───────────────────────────────────────────
        self.product_group_embedding = tf.keras.layers.Embedding(
            input_dim=num_product_groups + 1,
            output_dim=cat_emb_dim,
            embeddings_initializer="glorot_uniform",
            name="product_group_embedding",
        )

        # ── Colour group embedding ────────────────────────────────────────────
        self.colour_group_embedding = tf.keras.layers.Embedding(
            input_dim=num_colour_groups + 1,
            output_dim=cat_emb_dim,
            embeddings_initializer="glorot_uniform",
            name="colour_group_embedding",
        )

        # ── Popularity signal projection ──────────────────────────────────────
        self.pop_norm = tf.keras.layers.LayerNormalization(name="pop_norm")
        self.pop_proj = tf.keras.layers.Dense(
            cat_emb_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="pop_proj",
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
                - 'article_id_idx'         : int32 Tensor (batch,)
                - 'product_group_name_idx' : int32 Tensor (batch,)
                - 'colour_group_name_idx'  : int32 Tensor (batch,)
                - 'popularity_over_time'   : float32 Tensor (batch,)
            training: Boolean flag for dropout / batch-norm behaviour.

        Returns:
            tf.Tensor of shape (batch, embedding_dim), L2-normalised.
        """
        # ── Item identity embedding ───────────────────────────────────────────
        item_idx = tf.cast(inputs["article_id_idx"], tf.int32)
        item_emb = self.item_embedding(item_idx)                  # (B, item_emb_dim)

        # ── Categorical side embeddings ───────────────────────────────────────
        pg_idx = tf.cast(inputs["product_group_name_idx"], tf.int32)
        pg_emb = self.product_group_embedding(pg_idx)             # (B, cat_emb_dim)

        cg_idx = tf.cast(inputs["colour_group_name_idx"], tf.int32)
        cg_emb = self.colour_group_embedding(cg_idx)              # (B, cat_emb_dim)

        # ── Popularity projection ─────────────────────────────────────────────
        pop = tf.expand_dims(
            tf.cast(inputs["popularity_over_time"], tf.float32), axis=1
        )                                                           # (B, 1)
        pop_normed = self.pop_norm(pop, training=training)
        pop_emb = self.pop_proj(pop_normed, training=training)    # (B, cat_emb_dim)

        # ── Concatenate all branches ──────────────────────────────────────────
        x = tf.concat(
            [item_emb, pg_emb, cg_emb, pop_emb], axis=1
        )                                                           # (B, item_emb_dim + 3*cat_emb_dim)

        # ── MLP body ──────────────────────────────────────────────────────────
        x = self.dense1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)

        x = self.dense2(x, training=training)
        x = self.dropout2(x, training=training)

        x = self.output_proj(x)                                   # (B, embedding_dim)

        # ── L2-normalise ──────────────────────────────────────────────────────
        x = tf.math.l2_normalize(x, axis=-1)
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            {
                "num_items": self.num_items,
                "num_product_groups": self.num_product_groups,
                "num_colour_groups": self.num_colour_groups,
                "embedding_dim": self.embedding_dim,
                "item_emb_dim": self.item_emb_dim,
                "cat_emb_dim": self.cat_emb_dim,
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
    print("DAY 6: CANDIDATE TOWER ARCHITECTURE DEMO")
    print("=" * 70)

    tf.random.set_seed(42)
    batch_size = 8
    num_items = 1000

    tower = CandidateTower(
        num_items=num_items,
        num_product_groups=20,
        num_colour_groups=50,
        embedding_dim=64,
    )

    dummy_inputs = {
        "article_id_idx":          tf.random.uniform((batch_size,), 0, num_items, dtype=tf.int32),
        "product_group_name_idx":  tf.random.uniform((batch_size,), 0, 20, dtype=tf.int32),
        "colour_group_name_idx":   tf.random.uniform((batch_size,), 0, 50, dtype=tf.int32),
        "popularity_over_time":    tf.random.uniform((batch_size,), 1.0, 200.0, dtype=tf.float32),
    }

    output = tower(dummy_inputs, training=False)

    print(f"\n  Input batch size   : {batch_size}")
    print(f"  Output shape       : {output.shape}   (expected: ({batch_size}, 64))")
    norms = tf.norm(output, axis=-1).numpy()
    print(f"  L2 norms (all ≈1) : {norms.round(4)}")

    tower.summary()
    print("\nCandidate Tower demo completed successfully!")

import tensorflow as tf


class CandidateTower(tf.keras.Model):
    """Candidate tower for generating item embeddings."""

    def __init__(self, hidden_units=(64, 32), output_dim=32):
        super().__init__()

        self.dense_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in hidden_units
        ]

        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation=None
        )

    def call(self, embeddings):
        """Pass item embeddings through the candidate tower."""

        # Combine all categorical embeddings
        x = tf.concat(
            [embeddings[column] for column in embeddings],
            axis=-1
        )

        # Apply dense layers
        for layer in self.dense_layers:
            x = layer(x)

        # Generate final candidate representation
        return self.output_layer(x)
         main
