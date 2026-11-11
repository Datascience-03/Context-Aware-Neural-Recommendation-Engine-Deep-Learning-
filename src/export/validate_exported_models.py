"""
Day 6 — Validate exported Query Tower and Candidate Tower.

Validation:
1. Exported architecture JSON files can be loaded.
2. Exported H5 weights can be read and assigned.
3. Both reconstructed models produce 64-dimensional embeddings.
4. Outputs contain no NaN/Inf values.
5. Outputs are L2-normalized.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.model.query_tower import QueryTower
from src.model.candidate_tower import CandidateTower


EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_export"

QUERY_JSON = EXPORT_DIR / "query_tower.json"
QUERY_WEIGHTS = EXPORT_DIR / "query_tower.weights.h5"

CANDIDATE_JSON = EXPORT_DIR / "candidate_tower.json"
CANDIDATE_WEIGHTS = EXPORT_DIR / "candidate_tower.weights.h5"


def check_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found:\n{path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"{description} is empty:\n{path}"
        )


def build_validation_inputs(model_name: str):
    if model_name == "Query Tower":
        return {
            "customer_id_idx": tf.constant(
                [1, 2],
                dtype=tf.int32,
            ),
            "month_sin": tf.constant(
                [0.0, 0.5],
                dtype=tf.float32,
            ),
            "month_cos": tf.constant(
                [1.0, 0.5],
                dtype=tf.float32,
            ),
            "day_of_week_sin": tf.constant(
                [0.0, 0.5],
                dtype=tf.float32,
            ),
            "day_of_week_cos": tf.constant(
                [1.0, 0.5],
                dtype=tf.float32,
            ),
            "is_weekend": tf.constant(
                [0.0, 1.0],
                dtype=tf.float32,
            ),
            "days_since_last_purchase": tf.constant(
                [1.0, 5.0],
                dtype=tf.float32,
            ),
            "purchase_sequence": tf.constant(
                [1.0, 3.0],
                dtype=tf.float32,
            ),
        }

    if model_name == "Candidate Tower":
        return {
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
                [0.1, 0.5],
                dtype=tf.float32,
            ),
        }

    raise ValueError(f"Unknown model: {model_name}")


def load_json_architecture(
    json_path: Path,
    custom_class,
    model_name: str,
):
    print(f"  Loading {model_name} architecture...")

    with open(
        json_path,
        "r",
        encoding="utf-8",
    ) as f:
        architecture_json = f.read()

    model = keras.models.model_from_json(
        architecture_json,
        custom_objects={
            custom_class.__name__: custom_class,
        },
    )

    print("    Architecture JSON: PASS")

    inputs = build_validation_inputs(model_name)

    print("  Building model with validation inputs...")

    output = model(
        inputs,
        training=False,
    )

    print(
        f"    Initial output shape: {tuple(output.shape)}"
    )

    return model, inputs


def load_query_weights(
    model,
    weights_path: Path,
):
    print("  Reading exported Query Tower weights...")

    with h5py.File(weights_path, "r") as f:

        weights = {
            "user_embedding": f[
                "layers/embedding/vars/0"
            ][()],
            "context_norm": [
                f["context_norm/vars/0"][()],
                f["context_norm/vars/1"][()],
            ],
            "context_proj": [
                f["context_proj/vars/0"][()],
                f["context_proj/vars/1"][()],
            ],
            "dense1": [
                f["dense1/vars/0"][()],
                f["dense1/vars/1"][()],
            ],
            "bn1": [
                f["bn1/vars/0"][()],
                f["bn1/vars/1"][()],
                f["bn1/vars/2"][()],
                f["bn1/vars/3"][()],
            ],
            "dense2": [
                f["dense2/vars/0"][()],
                f["dense2/vars/1"][()],
            ],
            "output_proj": [
                f["layers/dense_3/vars/0"][()],
                f["layers/dense_3/vars/1"][()],
            ],
        }

    print("    H5 weight extraction: PASS")

    # Explicit assignment avoids Keras 3 H5 name-matching issues.
    model.user_embedding.set_weights(
        [weights["user_embedding"]]
    )

    model.context_norm.set_weights(
        weights["context_norm"]
    )

    model.context_proj.set_weights(
        weights["context_proj"]
    )

    model.dense1.set_weights(
        weights["dense1"]
    )

    model.bn1.set_weights(
        weights["bn1"]
    )

    model.dense2.set_weights(
        weights["dense2"]
    )

    model.output_proj.set_weights(
        weights["output_proj"]
    )

    print("    User embedding: PASS")
    print("    Context normalization: PASS")
    print("    Context projection: PASS")
    print("    Dense1: PASS")
    print("    Batch normalization: PASS")
    print("    Dense2: PASS")
    print("    Output projection: PASS")

    return model


def load_candidate_weights(
    model,
    weights_path: Path,
):
    print("  Reading exported Candidate Tower weights...")

    with h5py.File(weights_path, "r") as f:

        weights = {
            "item_embedding": f[
                "item_embedding/vars/0"
            ][()],
            "product_group_embedding": f[
                "layers/embedding_1/vars/0"
            ][()],
            "colour_group_embedding": f[
                "colour_group_embedding/vars/0"
            ][()],
            "pop_norm": [
                f[
                    "layers/layer_normalization/vars/0"
                ][()],
                f[
                    "layers/layer_normalization/vars/1"
                ][()],
            ],
            "pop_proj": [
                f["layers/dense/vars/0"][()],
                f["layers/dense/vars/1"][()],
            ],
            "dense1": [
                f["dense1/vars/0"][()],
                f["dense1/vars/1"][()],
            ],
            "bn1": [
                f["bn1/vars/0"][()],
                f["bn1/vars/1"][()],
                f["bn1/vars/2"][()],
                f["bn1/vars/3"][()],
            ],
            "dense2": [
                f["dense2/vars/0"][()],
                f["dense2/vars/1"][()],
            ],
            "output_proj": [
                f["layers/dense_3/vars/0"][()],
                f["layers/dense_3/vars/1"][()],
            ],
        }

    print("    H5 weight extraction: PASS")

    model.item_embedding.set_weights(
        [weights["item_embedding"]]
    )

    model.product_group_embedding.set_weights(
        [weights["product_group_embedding"]]
    )

    model.colour_group_embedding.set_weights(
        [weights["colour_group_embedding"]]
    )

    model.pop_norm.set_weights(
        weights["pop_norm"]
    )

    model.pop_proj.set_weights(
        weights["pop_proj"]
    )

    model.dense1.set_weights(
        weights["dense1"]
    )

    model.bn1.set_weights(
        weights["bn1"]
    )

    model.dense2.set_weights(
        weights["dense2"]
    )

    model.output_proj.set_weights(
        weights["output_proj"]
    )

    print("    Item embedding: PASS")
    print("    Product group embedding: PASS")
    print("    Colour group embedding: PASS")
    print("    Popularity normalization: PASS")
    print("    Popularity projection: PASS")
    print("    Dense1: PASS")
    print("    Batch normalization: PASS")
    print("    Dense2: PASS")
    print("    Output projection: PASS")

    return model


def validate_output(
    model,
    inputs,
    model_name: str,
):
    print(f"  Running {model_name} inference...")

    output = model(
        inputs,
        training=False,
    ).numpy()

    print(
        f"    Output shape: {output.shape}"
    )

    if output.shape != (2, 64):
        raise ValueError(
            f"{model_name} output shape mismatch.\n"
            f"Expected: (2, 64)\n"
            f"Actual:   {output.shape}"
        )

    print("    Output shape: PASS")

    if not np.isfinite(output).all():
        raise ValueError(
            f"{model_name} output contains NaN/Inf."
        )

    print("    NaN/Inf check: PASS")

    norms = np.linalg.norm(
        output,
        axis=1,
    )

    print(
        "    L2 norms:",
        np.round(norms, 6),
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{model_name} output is not L2-normalized."
        )

    print("    L2 normalization: PASS")

    return output


def main():
    print()
    print("=" * 75)
    print("DAY 6: EXPORTED MODEL LOADING + VALIDATION")
    print("=" * 75)

    print()
    print("Export directory:")
    print(f"  {EXPORT_DIR}")

    print()
    print("[1/6] Checking exported files...")

    check_file(
        QUERY_JSON,
        "Query Tower architecture",
    )

    check_file(
        QUERY_WEIGHTS,
        "Query Tower weights",
    )

    check_file(
        CANDIDATE_JSON,
        "Candidate Tower architecture",
    )

    check_file(
        CANDIDATE_WEIGHTS,
        "Candidate Tower weights",
    )

    print("  Query Tower architecture: PASS")
    print("  Query Tower weights: PASS")
    print("  Candidate Tower architecture: PASS")
    print("  Candidate Tower weights: PASS")

    print()
    print(
        "[2/6] Loading Query Tower from exported architecture..."
    )

    query_model, query_inputs = load_json_architecture(
        QUERY_JSON,
        QueryTower,
        "Query Tower",
    )

    query_model = load_query_weights(
        query_model,
        QUERY_WEIGHTS,
    )

    print("  Query Tower loading: PASS")

    print()
    print(
        "[3/6] Loading Candidate Tower from exported architecture..."
    )

    candidate_model, candidate_inputs = load_json_architecture(
        CANDIDATE_JSON,
        CandidateTower,
        "Candidate Tower",
    )

    candidate_model = load_candidate_weights(
        candidate_model,
        CANDIDATE_WEIGHTS,
    )

    print("  Candidate Tower loading: PASS")

    print()
    print("[4/6] Validating Query Tower inference...")

    query_output = validate_output(
        query_model,
        query_inputs,
        "Query Tower",
    )

    print("  Query Tower inference: PASS")

    print()
    print("[5/6] Validating Candidate Tower inference...")

    candidate_output = validate_output(
        candidate_model,
        candidate_inputs,
        "Candidate Tower",
    )

    print("  Candidate Tower inference: PASS")

    print()
    print("[6/6] Final export validation...")

    query_norms = np.linalg.norm(
        query_output,
        axis=1,
    )

    candidate_norms = np.linalg.norm(
        candidate_output,
        axis=1,
    )

    if query_output.shape != (2, 64):
        raise ValueError(
            "Query Tower final validation failed."
        )

    if candidate_output.shape != (2, 64):
        raise ValueError(
            "Candidate Tower final validation failed."
        )

    if not np.isfinite(query_output).all():
        raise ValueError(
            "Query Tower contains invalid values."
        )

    if not np.isfinite(candidate_output).all():
        raise ValueError(
            "Candidate Tower contains invalid values."
        )

    if not np.allclose(
        query_norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Query Tower normalization failed."
        )

    if not np.allclose(
        candidate_norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Candidate Tower normalization failed."
        )

    print("  Query Tower: PASS")
    print("  Candidate Tower: PASS")
    print("  Architecture loading: PASS")
    print("  Weight loading: PASS")
    print("  Inference validation: PASS")
    print("  Output dimensions: PASS")
    print("  NaN/Inf validation: PASS")
    print("  L2 normalization validation: PASS")

    print()
    print("=" * 75)
    print(
        "DAY 6 EXPORTED MODEL LOADING + VALIDATION PASSED"
    )
    print("=" * 75)

    print()
    print("Validated exports:")
    print(f"  Query architecture: {QUERY_JSON}")
    print(f"  Query weights:      {QUERY_WEIGHTS}")
    print(f"  Candidate architecture: {CANDIDATE_JSON}")
    print(f"  Candidate weights:      {CANDIDATE_WEIGHTS}")

    print()
    print("Query Tower output:")
    print(f"  Shape: {query_output.shape}")
    print(
        f"  L2 norms: {np.round(query_norms, 6)}"
    )

    print()
    print("Candidate Tower output:")
    print(f"  Shape: {candidate_output.shape}")
    print(
        f"  L2 norms: {np.round(candidate_norms, 6)}"
    )

    print()
    print("Day 6 complete.")


if __name__ == "__main__":
    main()