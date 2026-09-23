"""
Day 4: Export trained Query Tower weights.

Source checkpoint:
    data/processed/model/best_weights.weights.h5

The source checkpoint contains the complete Two-Tower model.
This script extracts only the trained Query Tower weights and
saves them as a standalone Query Tower weights file.

Output:
    data/processed/model_export/query_tower.weights.h5
"""

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_WEIGHTS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model"
    / "best_weights.weights.h5"
)

OUTPUT_WEIGHTS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_export"
    / "query_tower.weights.h5"
)


# -------------------------------------------------------------------
# Import actual Query Tower used during training
# -------------------------------------------------------------------

from src.model.query_tower import QueryTower


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Real training pipeline:
# max customer_id_idx = 51527
# num_users = max + 1 = 51528
#
# QueryTower itself creates:
# Embedding(input_dim=num_users + 1)
# therefore:
# 51528 + 1 = 51529 embedding rows.

NUM_USERS = 51528
EMBEDDING_DIM = 64
USER_EMB_DIM = 32
CONTEXT_PROJ_DIM = 16
HIDDEN_DIM = 128
DROPOUT_RATE = 0.10


# -------------------------------------------------------------------
# Helper: create model and build variables
# -------------------------------------------------------------------

def build_query_tower():
    tower = QueryTower(
        num_users=NUM_USERS,
        embedding_dim=EMBEDDING_DIM,
        user_emb_dim=USER_EMB_DIM,
        context_proj_dim=CONTEXT_PROJ_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    )

    dummy_inputs = {
        "customer_id_idx": tf.constant([1, 2], dtype=tf.int32),
        "month_sin": tf.constant([0.0, 0.5], dtype=tf.float32),
        "month_cos": tf.constant([1.0, 0.5], dtype=tf.float32),
        "day_of_week_sin": tf.constant([0.0, 0.7], dtype=tf.float32),
        "day_of_week_cos": tf.constant([1.0, 0.7], dtype=tf.float32),
        "is_weekend": tf.constant([0.0, 1.0], dtype=tf.float32),
        "days_since_last_purchase": tf.constant(
            [1.0, 5.0], dtype=tf.float32
        ),
        "purchase_sequence": tf.constant(
            [1.0, 10.0], dtype=tf.float32
        ),
    }

    # Build all layer variables.
    _ = tower(dummy_inputs, training=False)

    return tower, dummy_inputs


# -------------------------------------------------------------------
# Helper: read dataset from H5 checkpoint
# -------------------------------------------------------------------

def read_h5_array(h5_file, path):
    if path not in h5_file:
        raise KeyError(f"Missing checkpoint dataset: {path}")

    return np.array(h5_file[path])


# -------------------------------------------------------------------
# Extract Query Tower weights from full Two-Tower checkpoint
# -------------------------------------------------------------------

def extract_query_weights():
    print("\n" + "=" * 75)
    print("DAY 4: QUERY TOWER TRAINED WEIGHTS EXPORT")
    print("=" * 75)

    print(f"\nSource checkpoint:")
    print(f"  {SOURCE_WEIGHTS}")

    print(f"\nOutput weights:")
    print(f"  {OUTPUT_WEIGHTS}")

    if not SOURCE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found:\n{SOURCE_WEIGHTS}"
        )

    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Build standalone Query Tower
    # ---------------------------------------------------------------

    print("\n[1/7] Building standalone Query Tower...")

    tower, dummy_inputs = build_query_tower()

    print("  Query Tower creation: PASS")
    print(f"  Output dimension     : {EMBEDDING_DIM}")
    print(f"  User embedding dim   : {USER_EMB_DIM}")
    print(f"  Context projection   : {CONTEXT_PROJ_DIM}")
    print(f"  Hidden dimension     : {HIDDEN_DIM}")
    print(f"  Dropout              : {DROPOUT_RATE}")

    # ---------------------------------------------------------------
    # Read Query Tower weights from full checkpoint
    # ---------------------------------------------------------------

    print("\n[2/7] Reading Query Tower weights from checkpoint...")

    with h5py.File(SOURCE_WEIGHTS, "r") as h5:

        checkpoint_paths = {
            "user_embedding": (
                "layers/query_tower/layers/embedding/vars/0"
            ),
            "context_norm": [
                "layers/query_tower/context_norm/vars/0",
                "layers/query_tower/context_norm/vars/1",
            ],
            "context_proj": [
                "layers/query_tower/context_proj/vars/0",
                "layers/query_tower/context_proj/vars/1",
            ],
            "dense1": [
                "layers/query_tower/dense1/vars/0",
                "layers/query_tower/dense1/vars/1",
            ],
            "bn1": [
                "layers/query_tower/bn1/vars/0",
                "layers/query_tower/bn1/vars/1",
                "layers/query_tower/bn1/vars/2",
                "layers/query_tower/bn1/vars/3",
            ],
            "dense2": [
                "layers/query_tower/dense2/vars/0",
                "layers/query_tower/dense2/vars/1",
            ],
            "output_proj": [
                "layers/query_tower/layers/dense_3/vars/0",
                "layers/query_tower/layers/dense_3/vars/1",
            ],
        }

        weights = {}

        weights["user_embedding"] = read_h5_array(
            h5,
            checkpoint_paths["user_embedding"],
        )

        for layer_name in [
            "context_norm",
            "context_proj",
            "dense1",
            "bn1",
            "dense2",
            "output_proj",
        ]:
            weights[layer_name] = [
                read_h5_array(h5, path)
                for path in checkpoint_paths[layer_name]
            ]

    print("  Checkpoint extraction: PASS")

    # ---------------------------------------------------------------
    # Validate checkpoint shapes
    # ---------------------------------------------------------------

    print("\n[3/7] Validating extracted weight shapes...")

    expected_shapes = {
        "user_embedding": (51529, 32),
        "context_norm": [(7,), (7,)],
        "context_proj": [(7, 16), (16,)],
        "dense1": [(48, 128), (128,)],
        "bn1": [(128,), (128,), (128,), (128,)],
        "dense2": [(128, 64), (64,)],
        "output_proj": [(64, 64), (64,)],
    }

    actual_shapes = {
        "user_embedding": weights["user_embedding"].shape,
        "context_norm": [x.shape for x in weights["context_norm"]],
        "context_proj": [x.shape for x in weights["context_proj"]],
        "dense1": [x.shape for x in weights["dense1"]],
        "bn1": [x.shape for x in weights["bn1"]],
        "dense2": [x.shape for x in weights["dense2"]],
        "output_proj": [x.shape for x in weights["output_proj"]],
    }

    for name in expected_shapes:
        if actual_shapes[name] != expected_shapes[name]:
            raise ValueError(
                f"{name} shape mismatch.\n"
                f"Expected: {expected_shapes[name]}\n"
                f"Actual:   {actual_shapes[name]}"
            )

        print(f"  {name:18s}: {actual_shapes[name]} PASS")

    print("  All checkpoint shapes: PASS")

    # ---------------------------------------------------------------
    # Assign weights to standalone Query Tower
    # ---------------------------------------------------------------

    print("\n[4/7] Assigning trained weights to Query Tower...")

    tower.user_embedding.set_weights(
        [weights["user_embedding"]]
    )

    tower.context_norm.set_weights(
        weights["context_norm"]
    )

    tower.context_proj.set_weights(
        weights["context_proj"]
    )

    tower.dense1.set_weights(
        weights["dense1"]
    )

    tower.bn1.set_weights(
        weights["bn1"]
    )

    tower.dense2.set_weights(
        weights["dense2"]
    )

    tower.output_proj.set_weights(
        weights["output_proj"]
    )

    print("  User embedding: PASS")
    print("  Context normalization: PASS")
    print("  Context projection: PASS")
    print("  Dense1: PASS")
    print("  Batch normalization: PASS")
    print("  Dense2: PASS")
    print("  Output projection: PASS")

    # ---------------------------------------------------------------
    # Validate extracted model output
    # ---------------------------------------------------------------

    print("\n[5/7] Validating trained Query Tower output...")

    output_before_save = tower(
        dummy_inputs,
        training=False,
    ).numpy()

    print(f"  Output shape: {output_before_save.shape}")

    if output_before_save.shape != (2, 64):
        raise ValueError(
            f"Unexpected output shape: {output_before_save.shape}"
        )

    if not np.isfinite(output_before_save).all():
        raise ValueError(
            "Query Tower output contains NaN or Inf values."
        )

    norms = np.linalg.norm(
        output_before_save,
        axis=1,
    )

    print(f"  L2 norms: {np.round(norms, 6)}")

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(
            f"L2 normalization check failed: {norms}"
        )

    print("  Output shape: PASS")
    print("  NaN/Inf check: PASS")
    print("  L2 normalization: PASS")

    # ---------------------------------------------------------------
    # Save standalone Query Tower weights
    # ---------------------------------------------------------------

    print("\n[6/7] Saving standalone Query Tower weights...")

    tower.save_weights(OUTPUT_WEIGHTS)

    if not OUTPUT_WEIGHTS.exists():
        raise FileNotFoundError(
            "Exported weights file was not created."
        )

    file_size = OUTPUT_WEIGHTS.stat().st_size

    print("  Export: PASS")
    print(f"  File: {OUTPUT_WEIGHTS}")
    print(f"  Size: {file_size:,} bytes")

    if file_size <= 0:
        raise ValueError(
            "Exported weights file is empty."
        )

    # ---------------------------------------------------------------
    # Reload and compare
    # ---------------------------------------------------------------

    print("\n[7/7] Reloading exported weights and comparing outputs...")

    reloaded_tower, _ = build_query_tower()

    reloaded_tower.load_weights(
        OUTPUT_WEIGHTS
    )

    output_after_reload = reloaded_tower(
        dummy_inputs,
        training=False,
    ).numpy()

    output_difference = np.max(
        np.abs(
            output_before_save
            - output_after_reload
        )
    )

    print(
        f"  Maximum output difference: "
        f"{output_difference:.12f}"
    )

    if not np.allclose(
        output_before_save,
        output_after_reload,
        atol=1e-6,
    ):
        raise ValueError(
            "Reloaded Query Tower output does not match "
            "the extracted Query Tower output."
        )

    print("  Reload output comparison: PASS")

    # ---------------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------------

    print("\n" + "=" * 75)
    print("DAY 4 QUERY TOWER WEIGHTS EXPORT PASSED")
    print("=" * 75)

    print("\nExported file:")
    print(f"  {OUTPUT_WEIGHTS}")

    print("\nVerification:")
    print("  ✓ Trained Query Tower weights extracted")
    print("  ✓ Weight shapes validated")
    print("  ✓ Standalone Query Tower created")
    print("  ✓ Exported weights saved")
    print("  ✓ Exported weights successfully reloaded")
    print("  ✓ Output shape = (2, 64)")
    print("  ✓ No NaN/Inf values")
    print("  ✓ L2 normalization verified")
    print("  ✓ Original vs reloaded outputs match")

    print("\nDay 4 complete.\n")


if __name__ == "__main__":
    extract_query_weights()