import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.model.candidate_tower import CandidateTower


MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "model"
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_export"

ITEM_IDS_PATH = MODEL_DIR / "item_ids.npy"
OUTPUT_PATH = EXPORT_DIR / "candidate_tower.json"


def get_num_items():
    """Get the number of trained items from the exported item IDs."""
    item_ids = np.load(ITEM_IDS_PATH, allow_pickle=True)
    return len(item_ids)


def build_dummy_inputs():
    """Build inputs matching the actual CandidateTower interface."""
    return {
        "article_id_idx": tf.constant([0, 1], dtype=tf.int32),
        "product_group_name_idx": tf.constant([0, 1], dtype=tf.int32),
        "colour_group_name_idx": tf.constant([0, 1], dtype=tf.int32),
        "popularity_over_time": tf.constant(
            [10.0, 20.0],
            dtype=tf.float32,
        ),
    }


def export_architecture():
    print("=" * 70)
    print("MEMBER 1 - DAY 3")
    print("CANDIDATE TOWER ARCHITECTURE EXPORT")
    print("=" * 70)
    print()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    num_items = get_num_items()

    print(f"Number of trained items : {num_items}")

    model = CandidateTower(
        num_items=num_items,
        num_product_groups=20,
        num_colour_groups=50,
        embedding_dim=64,
        item_emb_dim=32,
        cat_emb_dim=8,
        hidden_dim=128,
        dropout_rate=0.10,
    )

    dummy_inputs = build_dummy_inputs()

    output = model(
        dummy_inputs,
        training=False,
    )

    if output.shape != (2, 64):
        raise RuntimeError(
            f"Unexpected Candidate Tower output shape: {output.shape}"
        )

    print("Candidate Tower creation : PASS")
    print(f"Output dimension         : {output.shape[-1]}")
    print("Item embedding dim       : 32")
    print("Category embedding dim   : 8")
    print("Hidden dimension         : 128")
    print("Dropout rate             : 0.1")

    architecture_json = model.to_json()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(architecture_json)

    print()
    print("Architecture export      : PASS")
    print(f"Saved to                 : {OUTPUT_PATH}")

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        exported_data = json.load(f)

    print("JSON parsing             : PASS")

    exported_class = exported_data.get("class_name")

    if exported_class != "CandidateTower":
        raise RuntimeError(
            f"Unexpected exported class: {exported_class}"
        )

    print(f"Exported class           : {exported_class}")

    config = exported_data.get("config", {})

    checks = {
        "num_items": num_items,
        "num_product_groups": 20,
        "num_colour_groups": 50,
        "embedding_dim": 64,
        "item_emb_dim": 32,
        "cat_emb_dim": 8,
        "hidden_dim": 128,
        "dropout_rate": 0.10,
    }

    for key, expected_value in checks.items():
        actual_value = config.get(key)

        if actual_value != expected_value:
            raise RuntimeError(
                f"{key} mismatch: "
                f"{actual_value} != {expected_value}"
            )

    print("Configuration check      : PASS")

    reloaded_model = tf.keras.models.model_from_json(
        architecture_json,
        custom_objects={
            "CandidateTower": CandidateTower
        },
    )

    reloaded_output = reloaded_model(
        dummy_inputs,
        training=False,
    )

    if reloaded_output.shape != (2, 64):
        raise RuntimeError(
            f"Unexpected reloaded output shape: "
            f"{reloaded_output.shape}"
        )

    print("JSON reload              : PASS")
    print(f"Reloaded output shape    : {reloaded_output.shape}")

    norms = tf.norm(
        reloaded_output,
        axis=-1,
    ).numpy()

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError(
            f"Reloaded output is not L2-normalized: {norms}"
        )

    print("L2 normalization check   : PASS")
    print(f"Reloaded L2 norms        : {norms.round(4)}")

    print()
    print("=" * 70)
    print("DAY 3 CANDIDATE TOWER ARCHITECTURE EXPORT PASSED")
    print("=" * 70)


if __name__ == "__main__":
    export_architecture()