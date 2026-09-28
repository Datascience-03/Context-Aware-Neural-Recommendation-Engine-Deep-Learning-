"""
Member 1 - Day 2
Query Tower Architecture Export

Exports the architecture/configuration of the Query Tower used by
the actual ModelTrainer training pipeline.

No trained weights are exported on Day 2.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.model.query_tower import QueryTower

MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "model"
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_export"

USER_IDS_PATH = MODEL_DIR / "user_ids.npy"
ARCHITECTURE_PATH = EXPORT_DIR / "query_tower.json"


def get_num_users():
    """Read the number of trained users from the exported user IDs."""
    if not USER_IDS_PATH.exists():
        raise FileNotFoundError(
            f"User ID file not found: {USER_IDS_PATH}"
        )

    user_ids = np.load(USER_IDS_PATH, allow_pickle=True)

    if user_ids.ndim != 1:
        raise ValueError(
            f"Expected 1-D user ID array, got shape {user_ids.shape}"
        )

    return int(len(user_ids))


def build_query_tower(num_users):
    """Create and build the same Query Tower architecture used in training."""

    model = QueryTower(
        num_users=num_users,
        embedding_dim=64,
        user_emb_dim=32,
        context_proj_dim=16,
        hidden_dim=128,
        dropout_rate=0.10,
    )

    dummy_batch_size = 2

    dummy_inputs = {
        "customer_id_idx": tf.constant(
            [1, 2], dtype=tf.int32
        ),
        "month_sin": tf.zeros(
            [dummy_batch_size], dtype=tf.float32
        ),
        "month_cos": tf.zeros(
            [dummy_batch_size], dtype=tf.float32
        ),
        "day_of_week_sin": tf.zeros(
            [dummy_batch_size], dtype=tf.float32
        ),
        "day_of_week_cos": tf.zeros(
            [dummy_batch_size], dtype=tf.float32
        ),
        "is_weekend": tf.zeros(
            [dummy_batch_size], dtype=tf.float32
        ),
        "days_since_last_purchase": tf.zeros(
            [dummy_batch_size], dtype=tf.float32
        ),
        "purchase_sequence": tf.zeros(
            [dummy_batch_size], dtype=tf.float32
        ),
    }

    output = model(dummy_inputs, training=False)

    if tuple(output.shape) != (dummy_batch_size, 64):
        raise ValueError(
            f"Unexpected Query Tower output shape: {output.shape}"
        )

    return model


def export_architecture():
    """Export and verify the Query Tower architecture."""

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEMBER 1 - DAY 2")
    print("QUERY TOWER ARCHITECTURE EXPORT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Determine real user vocabulary size
    # ---------------------------------------------------------
    num_users = get_num_users()

    print(f"\nNumber of trained users : {num_users}")

    # ---------------------------------------------------------
    # 2. Build actual Query Tower
    # ---------------------------------------------------------
    model = build_query_tower(num_users)

    print("Query Tower creation   : PASS")
    print(f"Output dimension       : {model.embedding_dim}")
    print(f"User embedding dim     : {model.user_emb_dim}")
    print(f"Context projection dim : {model.context_proj_dim}")
    print(f"Hidden dimension       : {model.hidden_dim}")
    print(f"Dropout rate           : {model.dropout_rate}")

    # ---------------------------------------------------------
    # 3. Export Keras architecture JSON
    # ---------------------------------------------------------
    architecture_json = model.to_json()

    with open(
        ARCHITECTURE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(architecture_json)

    print("\nArchitecture export    : PASS")
    print(f"Saved to               : {ARCHITECTURE_PATH}")

    # ---------------------------------------------------------
    # 4. Verify JSON can be parsed
    # ---------------------------------------------------------
    with open(
        ARCHITECTURE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        exported = json.load(file)

    if "class_name" not in exported:
        raise ValueError(
            "Exported JSON does not contain class_name."
        )

    if "config" not in exported:
        raise ValueError(
            "Exported JSON does not contain config."
        )

    print("JSON parsing           : PASS")
    print(f"Exported class         : {exported['class_name']}")

    # ---------------------------------------------------------
    # 5. Verify important architecture configuration
    # ---------------------------------------------------------
    config = exported["config"]

    expected_config = {
        "num_users": num_users,
        "embedding_dim": 64,
        "user_emb_dim": 32,
        "context_proj_dim": 16,
        "hidden_dim": 128,
        "dropout_rate": 0.10,
    }

    for key, expected_value in expected_config.items():
        actual_value = config.get(key)

        if actual_value != expected_value:
            raise ValueError(
                f"Architecture mismatch for {key}: "
                f"expected {expected_value}, got {actual_value}"
            )

    print("Configuration check   : PASS")

    # ---------------------------------------------------------
    # 6. Reload architecture from JSON
    # ---------------------------------------------------------
    restored_model = tf.keras.models.model_from_json(
        architecture_json,
        custom_objects={"QueryTower": QueryTower},
    )

    restored_output = restored_model(
        {
            "customer_id_idx": tf.constant(
                [1, 2], dtype=tf.int32
            ),
            "month_sin": tf.zeros([2], dtype=tf.float32),
            "month_cos": tf.zeros([2], dtype=tf.float32),
            "day_of_week_sin": tf.zeros([2], dtype=tf.float32),
            "day_of_week_cos": tf.zeros([2], dtype=tf.float32),
            "is_weekend": tf.zeros([2], dtype=tf.float32),
            "days_since_last_purchase": tf.zeros(
                [2], dtype=tf.float32
            ),
            "purchase_sequence": tf.zeros(
                [2], dtype=tf.float32
            ),
        },
        training=False,
    )

    if tuple(restored_output.shape) != (2, 64):
        raise ValueError(
            f"Reloaded model output shape is incorrect: "
            f"{restored_output.shape}"
        )

    print("JSON reload            : PASS")
    print(f"Reloaded output shape  : {tuple(restored_output.shape)}")

    print("\n" + "=" * 70)
    print("DAY 2 QUERY TOWER ARCHITECTURE EXPORT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    export_architecture()