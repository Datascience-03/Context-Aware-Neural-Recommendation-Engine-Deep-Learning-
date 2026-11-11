"""
Day 5: Export trained Candidate Tower weights.

Source checkpoint:
    data/processed/model/best_weights.weights.h5

The source checkpoint contains the complete Two-Tower model.
This script extracts only the trained Candidate Tower weights
and saves them as a standalone Candidate Tower weights file.

Output:
    data/processed/model_export/candidate_tower.weights.h5
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
    / "candidate_tower.weights.h5"
)


# -------------------------------------------------------------------
# Import actual Candidate Tower used during training
# -------------------------------------------------------------------

from src.model.candidate_tower import CandidateTower


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Real trained model configuration.

NUM_ITEMS = 19520
NUM_PRODUCT_GROUPS = 15
NUM_COLOUR_GROUPS = 51
EMBEDDING_DIM = 64
ITEM_EMB_DIM = 32
CAT_EMB_DIM = 8
HIDDEN_DIM = 128
DROPOUT_RATE = 0.10


# -------------------------------------------------------------------
# Helper: create model and build variables
# -------------------------------------------------------------------

def build_candidate_tower():

    tower = CandidateTower(
        num_items=NUM_ITEMS,
        num_product_groups=NUM_PRODUCT_GROUPS,
        num_colour_groups=NUM_COLOUR_GROUPS,
        embedding_dim=EMBEDDING_DIM,
        item_emb_dim=ITEM_EMB_DIM,
        cat_emb_dim=CAT_EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
    )

    dummy_inputs = {
        "article_id_idx": tf.constant(
            [1, 2],
            dtype=tf.int32,
        ),
        "product_group_name_idx": tf.constant(
            [1, 2],
            dtype=tf.int32,
        ),
        "colour_group_name_idx": tf.constant(
            [1, 2],
            dtype=tf.int32,
        ),
        "popularity_over_time": tf.constant(
            [10.0, 25.0],
            dtype=tf.float32,
        ),
    }

    # Build all layer variables.
    _ = tower(
        dummy_inputs,
        training=False,
    )

    return tower, dummy_inputs


# -------------------------------------------------------------------
# Helper: read dataset from H5 checkpoint
# -------------------------------------------------------------------

def read_h5_array(h5_file, path):

    if path not in h5_file:
        raise KeyError(
            f"Missing checkpoint dataset: {path}"
        )

    return np.array(
        h5_file[path]
    )


# -------------------------------------------------------------------
# Main export process
# -------------------------------------------------------------------

def extract_candidate_weights():

    print("\n" + "=" * 75)
    print("DAY 5: CANDIDATE TOWER TRAINED WEIGHTS EXPORT")
    print("=" * 75)

    print("\nSource checkpoint:")
    print(f"  {SOURCE_WEIGHTS}")

    print("\nOutput weights:")
    print(f"  {OUTPUT_WEIGHTS}")

    if not SOURCE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Source checkpoint not found:\n{SOURCE_WEIGHTS}"
        )

    OUTPUT_WEIGHTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # 1. Build standalone Candidate Tower
    # ---------------------------------------------------------------

    print(
        "\n[1/7] Building standalone Candidate Tower..."
    )

    tower, dummy_inputs = build_candidate_tower()

    print("  Candidate Tower creation: PASS")
    print(f"  Number of items       : {NUM_ITEMS}")
    print(f"  Product groups        : {NUM_PRODUCT_GROUPS}")
    print(f"  Colour groups         : {NUM_COLOUR_GROUPS}")
    print(f"  Output dimension      : {EMBEDDING_DIM}")
    print(f"  Item embedding dim    : {ITEM_EMB_DIM}")
    print(f"  Category embedding    : {CAT_EMB_DIM}")
    print(f"  Hidden dimension      : {HIDDEN_DIM}")
    print(f"  Dropout               : {DROPOUT_RATE}")

    # ---------------------------------------------------------------
    # 2. Read Candidate Tower weights from checkpoint
    # ---------------------------------------------------------------

    print(
        "\n[2/7] Reading Candidate Tower weights from checkpoint..."
    )

    with h5py.File(
        SOURCE_WEIGHTS,
        "r",
    ) as h5:

        checkpoint_paths = {

            "item_embedding": (
                "candidate_tower/item_embedding/vars/0"
            ),

            "product_group_embedding": (
                "candidate_tower/layers/embedding_1/vars/0"
            ),

            "colour_group_embedding": (
                "candidate_tower/colour_group_embedding/vars/0"
            ),

            "pop_norm": [
                "candidate_tower/layers/layer_normalization/vars/0",
                "candidate_tower/layers/layer_normalization/vars/1",
            ],

            "pop_proj": [
                "candidate_tower/layers/dense/vars/0",
                "candidate_tower/layers/dense/vars/1",
            ],

            "dense1": [
                "candidate_tower/dense1/vars/0",
                "candidate_tower/dense1/vars/1",
            ],

            "bn1": [
                "candidate_tower/bn1/vars/0",
                "candidate_tower/bn1/vars/1",
                "candidate_tower/bn1/vars/2",
                "candidate_tower/bn1/vars/3",
            ],

            "dense2": [
                "candidate_tower/dense2/vars/0",
                "candidate_tower/dense2/vars/1",
            ],

            "output_proj": [
                "candidate_tower/layers/dense_3/vars/0",
                "candidate_tower/layers/dense_3/vars/1",
            ],
        }

        weights = {}

        weights["item_embedding"] = read_h5_array(
            h5,
            checkpoint_paths["item_embedding"],
        )

        weights["product_group_embedding"] = read_h5_array(
            h5,
            checkpoint_paths["product_group_embedding"],
        )

        weights["colour_group_embedding"] = read_h5_array(
            h5,
            checkpoint_paths["colour_group_embedding"],
        )

        for layer_name in [
            "pop_norm",
            "pop_proj",
            "dense1",
            "bn1",
            "dense2",
            "output_proj",
        ]:

            weights[layer_name] = [
                read_h5_array(
                    h5,
                    path,
                )
                for path in checkpoint_paths[layer_name]
            ]

    print("  Checkpoint extraction: PASS")

    # ---------------------------------------------------------------
    # 3. Validate checkpoint weight shapes
    # ---------------------------------------------------------------

    print(
        "\n[3/7] Validating extracted weight shapes..."
    )

    expected_shapes = {

        "item_embedding":
            (NUM_ITEMS + 1, ITEM_EMB_DIM),

        "product_group_embedding":
            (NUM_PRODUCT_GROUPS + 1, CAT_EMB_DIM),

        "colour_group_embedding":
            (NUM_COLOUR_GROUPS + 1, CAT_EMB_DIM),

        "pop_norm":
            [(1,), (1,)],

        "pop_proj":
            [(1, CAT_EMB_DIM), (CAT_EMB_DIM,)],

        "dense1":
            [(56, HIDDEN_DIM), (HIDDEN_DIM,)],

        "bn1":
            [
                (HIDDEN_DIM,),
                (HIDDEN_DIM,),
                (HIDDEN_DIM,),
                (HIDDEN_DIM,),
            ],

        "dense2":
            [
                (HIDDEN_DIM, HIDDEN_DIM // 2),
                (HIDDEN_DIM // 2,),
            ],

        "output_proj":
            [
                (HIDDEN_DIM // 2, EMBEDDING_DIM),
                (EMBEDDING_DIM,),
            ],
    }

    actual_shapes = {

        "item_embedding":
            weights["item_embedding"].shape,

        "product_group_embedding":
            weights["product_group_embedding"].shape,

        "colour_group_embedding":
            weights["colour_group_embedding"].shape,

        "pop_norm":
            [
                x.shape
                for x in weights["pop_norm"]
            ],

        "pop_proj":
            [
                x.shape
                for x in weights["pop_proj"]
            ],

        "dense1":
            [
                x.shape
                for x in weights["dense1"]
            ],

        "bn1":
            [
                x.shape
                for x in weights["bn1"]
            ],

        "dense2":
            [
                x.shape
                for x in weights["dense2"]
            ],

        "output_proj":
            [
                x.shape
                for x in weights["output_proj"]
            ],
    }

    for name in expected_shapes:

        if actual_shapes[name] != expected_shapes[name]:

            raise ValueError(
                f"{name} shape mismatch.\n"
                f"Expected: {expected_shapes[name]}\n"
                f"Actual:   {actual_shapes[name]}"
            )

        print(
            f"  {name:25s}: "
            f"{actual_shapes[name]} PASS"
        )

    print(
        "  All checkpoint shapes: PASS"
    )

    # ---------------------------------------------------------------
    # 4. Assign trained weights
    # ---------------------------------------------------------------

    print(
        "\n[4/7] Assigning trained weights "
        "to Candidate Tower..."
    )

    tower.item_embedding.set_weights(
        [weights["item_embedding"]]
    )

    tower.product_group_embedding.set_weights(
        [weights["product_group_embedding"]]
    )

    tower.colour_group_embedding.set_weights(
        [weights["colour_group_embedding"]]
    )

    tower.pop_norm.set_weights(
        weights["pop_norm"]
    )

    tower.pop_proj.set_weights(
        weights["pop_proj"]
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

    print("  Item embedding: PASS")
    print("  Product group embedding: PASS")
    print("  Colour group embedding: PASS")
    print("  Popularity normalization: PASS")
    print("  Popularity projection: PASS")
    print("  Dense1: PASS")
    print("  Batch normalization: PASS")
    print("  Dense2: PASS")
    print("  Output projection: PASS")

    # ---------------------------------------------------------------
    # 5. Validate Candidate Tower output
    # ---------------------------------------------------------------

    print(
        "\n[5/7] Validating trained Candidate Tower output..."
    )

    output_before_save = tower(
        dummy_inputs,
        training=False,
    ).numpy()

    print(
        f"  Output shape: "
        f"{output_before_save.shape}"
    )

    if output_before_save.shape != (
        2,
        EMBEDDING_DIM,
    ):

        raise ValueError(
            f"Unexpected output shape: "
            f"{output_before_save.shape}"
        )

    if not np.isfinite(
        output_before_save
    ).all():

        raise ValueError(
            "Candidate Tower output contains "
            "NaN or Inf values."
        )

    norms = np.linalg.norm(
        output_before_save,
        axis=1,
    )

    print(
        f"  L2 norms: "
        f"{np.round(norms, 6)}"
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ):

        raise ValueError(
            f"L2 normalization check failed: "
            f"{norms}"
        )

    print("  Output shape: PASS")
    print("  NaN/Inf check: PASS")
    print("  L2 normalization: PASS")

    # ---------------------------------------------------------------
    # 6. Save standalone Candidate Tower weights
    # ---------------------------------------------------------------

    print(
        "\n[6/7] Saving standalone Candidate Tower weights..."
    )

    tower.save_weights(
        OUTPUT_WEIGHTS
    )

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
    # 7. Reload and compare
    # ---------------------------------------------------------------

    print(
        "\n[7/7] Reloading exported weights "
        "and comparing outputs..."
    )

    reloaded_tower, _ = build_candidate_tower()

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
        "  Maximum output difference: "
        f"{output_difference:.12f}"
    )

    if not np.allclose(
        output_before_save,
        output_after_reload,
        atol=1e-6,
    ):

        raise ValueError(
            "Reloaded Candidate Tower output "
            "does not match the extracted "
            "Candidate Tower output."
        )

    print(
        "  Reload output comparison: PASS"
    )

    # ---------------------------------------------------------------
    # Final verification
    # ---------------------------------------------------------------

    print("\n" + "=" * 75)
    print(
        "DAY 5 CANDIDATE TOWER WEIGHTS EXPORT PASSED"
    )
    print("=" * 75)

    print("\nExported file:")
    print(
        f"  {OUTPUT_WEIGHTS}"
    )

    print("\nVerification:")
    print(
        "  ✓ Trained Candidate Tower weights extracted"
    )
    print(
        "  ✓ Weight shapes validated"
    )
    print(
        "  ✓ Standalone Candidate Tower created"
    )
    print(
        "  ✓ Exported weights saved"
    )
    print(
        "  ✓ Exported weights successfully reloaded"
    )
    print(
        "  ✓ Output shape = (2, 64)"
    )
    print(
        "  ✓ No NaN/Inf values"
    )
    print(
        "  ✓ L2 normalization verified"
    )
    print(
        "  ✓ Original vs reloaded outputs match"
    )

    print("\nDay 5 complete.\n")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    extract_candidate_weights()