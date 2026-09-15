
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

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

logger = logging.getLogger(__name__)


class CandidateTower(tf.keras.Model):
    """
    Item Feature Encoder Tower for the Two-Tower Retrieval Model.

    Architecture:
        article_id → Embedding(num_items, item_emb_dim)
        product_group_idx → Embedding(num_product_groups, cat_emb_dim)
        colour_group_idx  → Embedding(num_colour_groups, cat_emb_dim)
        popularity_over_time → Dense(cat_emb_dim, relu)
            ↓
        concat([item_emb, product_emb, colour_emb, pop_proj])
            ↓
        Dense(hidden_dim, relu) → BatchNorm → Dropout
            ↓
        Dense(hidden_dim // 2, relu) → Dropout
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
            embedding_dim: Output embedding dimensionality.
            item_emb_dim: Dimensionality of item identity embedding.
            cat_emb_dim: Dimensionality of categorical embeddings.
            hidden_dim: Hidden size of the main MLP.
            dropout_rate: Dropout probability.
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

        # ------------------------------------------------------------------
        # Item identity embedding
        # ------------------------------------------------------------------
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=num_items + 1,
            output_dim=item_emb_dim,
            embeddings_initializer="glorot_uniform",
            embeddings_regularizer=tf.keras.regularizers.L2(1e-5),
            name="item_embedding",
        )

        # ------------------------------------------------------------------
        # Product group embedding
        # ------------------------------------------------------------------
        self.product_group_embedding = tf.keras.layers.Embedding(
            input_dim=num_product_groups + 1,
            output_dim=cat_emb_dim,
            embeddings_initializer="glorot_uniform",
            name="product_group_embedding",
        )

        # ------------------------------------------------------------------
        # Colour group embedding
        # ------------------------------------------------------------------
        self.colour_group_embedding = tf.keras.layers.Embedding(
            input_dim=num_colour_groups + 1,
            output_dim=cat_emb_dim,
            embeddings_initializer="glorot_uniform",
            name="colour_group_embedding",
        )

        # ------------------------------------------------------------------
        # Popularity signal projection
        # ------------------------------------------------------------------
        self.pop_norm = tf.keras.layers.LayerNormalization(
            name="pop_norm"
        )

        self.pop_proj = tf.keras.layers.Dense(
            cat_emb_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="pop_proj",
        )

        # ------------------------------------------------------------------
        # Main MLP body
        # ------------------------------------------------------------------
        self.dense1 = tf.keras.layers.Dense(
            hidden_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="dense1",
        )

        self.bn1 = tf.keras.layers.BatchNormalization(
            name="bn1"
        )

        self.dropout1 = tf.keras.layers.Dropout(
            dropout_rate,
            name="dropout1"
        )

        self.dense2 = tf.keras.layers.Dense(
            hidden_dim // 2,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(1e-5),
            name="dense2",
        )

        self.dropout2 = tf.keras.layers.Dropout(
            dropout_rate,
            name="dropout2"
        )

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
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
                - article_id_idx
                - product_group_name_idx
                - colour_group_name_idx
                - popularity_over_time

            training: Controls dropout and batch-normalisation behaviour.

        Returns:
            L2-normalised candidate embeddings.
        """

        # ------------------------------------------------------------------
        # Item identity embedding
        # ------------------------------------------------------------------
        item_idx = tf.cast(
            inputs["article_id_idx"],
            tf.int32
        )

        item_emb = self.item_embedding(item_idx)

        # ------------------------------------------------------------------
        # Product group embedding
        # ------------------------------------------------------------------
        pg_idx = tf.cast(
            inputs["product_group_name_idx"],
            tf.int32
        )

        pg_emb = self.product_group_embedding(pg_idx)

        # ------------------------------------------------------------------
        # Colour group embedding
        # ------------------------------------------------------------------
        cg_idx = tf.cast(
            inputs["colour_group_name_idx"],
            tf.int32
        )

        cg_emb = self.colour_group_embedding(cg_idx)

        # ------------------------------------------------------------------
        # Popularity projection
        # ------------------------------------------------------------------
        pop = tf.expand_dims(
            tf.cast(
                inputs["popularity_over_time"],
                tf.float32
            ),
            axis=1,
        )

        pop_normed = self.pop_norm(
            pop,
            training=training
        )

        pop_emb = self.pop_proj(
            pop_normed,
            training=training
        )

        # ------------------------------------------------------------------
        # Concatenate all feature branches
        # ------------------------------------------------------------------
        x = tf.concat(
            [
                item_emb,
                pg_emb,
                cg_emb,
                pop_emb,
            ],
            axis=1,
        )

        # ------------------------------------------------------------------
        # MLP body
        # ------------------------------------------------------------------
        x = self.dense1(
            x,
            training=training
        )

        x = self.bn1(
            x,
            training=training
        )

        x = self.dropout1(
            x,
            training=training
        )

        x = self.dense2(
            x,
            training=training
        )

        x = self.dropout2(
            x,
            training=training
        )

        # ------------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------------
        x = self.output_proj(x)

        # ------------------------------------------------------------------
        # L2 normalisation
        # ------------------------------------------------------------------
        x = tf.math.l2_normalize(
            x,
            axis=-1
        )

        return x

    def get_config(self) -> dict:
        """Return model configuration for serialization."""

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


# ============================================================================
# Standalone demo
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("CANDIDATE TOWER ARCHITECTURE DEMO")
    print("=" * 70)

    tf.random.set_seed(42)

    batch_size = 8
    num_items = 1000

    # ------------------------------------------------------------------
    # Create Candidate Tower
    # ------------------------------------------------------------------
    tower = CandidateTower(
        num_items=num_items,
        num_product_groups=20,
        num_colour_groups=50,
        embedding_dim=64,
    )

    # ------------------------------------------------------------------
    # Create synthetic input batch
    # ------------------------------------------------------------------
    dummy_inputs = {
        "article_id_idx": tf.random.uniform(
            (batch_size,),
            0,
            num_items,
            dtype=tf.int32,
        ),
        "product_group_name_idx": tf.random.uniform(
            (batch_size,),
            0,
            20,
            dtype=tf.int32,
        ),
        "colour_group_name_idx": tf.random.uniform(
            (batch_size,),
            0,
            50,
            dtype=tf.int32,
        ),
        "popularity_over_time": tf.random.uniform(
            (batch_size,),
            1.0,
            200.0,
            dtype=tf.float32,
        ),
    }

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    output = tower(
        dummy_inputs,
        training=False,
    )

    print(f"\n  Input batch size : {batch_size}")
    print(
        f"  Output shape     : {output.shape} "
        f"(expected: ({batch_size}, 64))"
    )

    # ------------------------------------------------------------------
    # Check L2 norms
    # ------------------------------------------------------------------
    norms = tf.norm(
        output,
        axis=-1
    ).numpy()

    print(
        f"  L2 norms (all ≈1): {norms.round(4)}"
    )

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------
    tower.summary()

    print("\nCandidate Tower demo completed successfully!")