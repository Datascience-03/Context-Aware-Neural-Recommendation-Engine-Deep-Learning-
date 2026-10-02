import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "model"

ITEM_IDS_FILE = MODEL_DIR / "item_ids.npy"
ITEM_EMBEDDINGS_FILE = MODEL_DIR / "item_embeddings.npy"
MANIFEST_FILE = MODEL_DIR / "item_embedding_manifest.json"

EXPECTED_DIMENSION = 64
EXPECTED_DTYPE = np.float32
EXPECTED_ITEM_COUNT = 19519


def validate_item_embeddings():
    print("=" * 70)
    print("ITEM EMBEDDING VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Check required files
    # ---------------------------------------------------------
    print("\n[1] Checking required files...")

    required_files = [
        ITEM_IDS_FILE,
        ITEM_EMBEDDINGS_FILE,
        MANIFEST_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        print(f"PASS: {file_path.name}")

    # ---------------------------------------------------------
    # 2. Load files
    # ---------------------------------------------------------
    print("\n[2] Loading item IDs and embeddings...")

    item_ids = np.load(ITEM_IDS_FILE, allow_pickle=True)
    item_embeddings = np.load(ITEM_EMBEDDINGS_FILE)

    print(f"Item IDs shape        : {item_ids.shape}")
    print(f"Item embeddings shape : {item_embeddings.shape}")

    # ---------------------------------------------------------
    # 3. Validate item count
    # ---------------------------------------------------------
    print("\n[3] Validating item count...")

    assert len(item_ids) == EXPECTED_ITEM_COUNT, (
        f"Expected {EXPECTED_ITEM_COUNT} item IDs, "
        f"found {len(item_ids)}"
    )

    assert len(item_embeddings) == EXPECTED_ITEM_COUNT, (
        f"Expected {EXPECTED_ITEM_COUNT} embeddings, "
        f"found {len(item_embeddings)}"
    )

    print(f"PASS: {EXPECTED_ITEM_COUNT} items")

    # ---------------------------------------------------------
    # 4. Validate embedding shape
    # ---------------------------------------------------------
    print("\n[4] Validating embedding dimension...")

    assert item_embeddings.ndim == 2, (
        f"Expected 2D embeddings, got {item_embeddings.ndim}D"
    )

    assert item_embeddings.shape[1] == EXPECTED_DIMENSION, (
        f"Expected dimension {EXPECTED_DIMENSION}, "
        f"found {item_embeddings.shape[1]}"
    )

    print(f"PASS: embedding dimension = {EXPECTED_DIMENSION}")

    # ---------------------------------------------------------
    # 5. Validate dtype
    # ---------------------------------------------------------
    print("\n[5] Validating dtype...")

    assert item_embeddings.dtype == EXPECTED_DTYPE, (
        f"Expected dtype {EXPECTED_DTYPE}, "
        f"found {item_embeddings.dtype}"
    )

    print(f"PASS: dtype = {item_embeddings.dtype}")

    # ---------------------------------------------------------
    # 6. Validate ID/embedding alignment
    # ---------------------------------------------------------
    print("\n[6] Validating ID/embedding alignment...")

    assert len(item_ids) == len(item_embeddings)

    print("PASS: ID count matches embedding count")

    # ---------------------------------------------------------
    # 7. Validate unique IDs
    # ---------------------------------------------------------
    print("\n[7] Validating unique item IDs...")

    unique_count = len(np.unique(item_ids))

    assert unique_count == len(item_ids), (
        f"Duplicate IDs found: {len(item_ids) - unique_count}"
    )

    print(f"PASS: all {unique_count} item IDs are unique")

    # ---------------------------------------------------------
    # 8. Validate NaN / Inf
    # ---------------------------------------------------------
    print("\n[8] Checking NaN and Inf values...")

    assert not np.isnan(item_embeddings).any(), (
        "NaN values found in item embeddings"
    )

    assert not np.isinf(item_embeddings).any(), (
        "Inf values found in item embeddings"
    )

    print("PASS: no NaN values")
    print("PASS: no Inf values")

    # ---------------------------------------------------------
    # 9. Validate L2 normalization
    # ---------------------------------------------------------
    print("\n[9] Validating L2 normalization...")

    norms = np.linalg.norm(item_embeddings, axis=1)

    print(f"Min norm  : {norms.min():.8f}")
    print(f"Max norm  : {norms.max():.8f}")
    print(f"Mean norm : {norms.mean():.8f}")

    assert np.allclose(norms, 1.0, atol=1e-5), (
        "Embeddings are not L2 normalized"
    )

    print("PASS: all embedding norms are approximately 1.0")

    # ---------------------------------------------------------
    # 10. Validate manifest
    # ---------------------------------------------------------
    print("\n[10] Validating manifest...")

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["embedding_file"] == "item_embeddings.npy"
    assert manifest["item_id_file"] == "item_ids.npy"
    assert manifest["num_items"] == EXPECTED_ITEM_COUNT
    assert manifest["embedding_dimension"] == EXPECTED_DIMENSION
    assert manifest["dtype"] == "float32"
    assert manifest["normalized"] is True
    assert manifest["model"] == "CandidateTower"

    print("PASS: manifest is valid")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("ITEM EMBEDDING VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    validate_item_embeddings()