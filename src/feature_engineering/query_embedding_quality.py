from pathlib import Path

import pandas as pd
import tensorflow as tf

from query_tower import (
    QueryTower,
    load_embedding_config,
    USER_CATEGORICAL_FEATURES,
    USER_NUMERICAL_FEATURES,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "query_tower_inputs.csv"
)


# ============================================================
# LOAD REAL QUERY TOWER DATA
# ============================================================

def load_query_inputs(sample_size=32):
    """
    Load real Query Tower inputs from the processed dataset.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Query Tower input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        nrows=sample_size
    )

    inputs = {}

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    for feature in USER_CATEGORICAL_FEATURES:

        column_name = f"{feature}_idx"

        if column_name not in df.columns:
            raise KeyError(
                f"Required column '{column_name}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        inputs[feature] = tf.convert_to_tensor(
            df[column_name].values,
            dtype=tf.int32
        )

    # --------------------------------------------------------
    # NUMERICAL / CONTEXTUAL FEATURES
    # --------------------------------------------------------

    for feature in USER_NUMERICAL_FEATURES:

        if feature not in df.columns:
            raise KeyError(
                f"Required feature '{feature}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        inputs[feature] = tf.convert_to_tensor(
            df[feature].values,
            dtype=tf.float32
        )

    return df, inputs


# ============================================================
# QUALITY TEST
# ============================================================

def test_query_embedding_quality():

    print("=" * 60)
    print("QUERY EMBEDDING QUALITY TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: LOAD REAL DATA
    # --------------------------------------------------------

    print("\nLoading real Query Tower data...")

    df, inputs = load_query_inputs(
        sample_size=32
    )

    print(f"Loaded records: {len(df)}")

    # --------------------------------------------------------
    # STEP 2: CREATE QUERY TOWER
    # --------------------------------------------------------

    print("\nLoading embedding configuration...")

    embedding_config = load_embedding_config()

    model = QueryTower(
        embedding_config=embedding_config,
        output_dim=64
    )

    print("Query Tower created successfully.")

    # --------------------------------------------------------
    # STEP 3: GENERATE EMBEDDINGS
    # --------------------------------------------------------

    print("\nGenerating Query embeddings...")

    embeddings_1 = model(
        inputs,
        training=False
    )

    print(
        "Embedding shape:",
        embeddings_1.shape
    )

    # --------------------------------------------------------
    # CHECK 1: EMBEDDING DIMENSION
    # --------------------------------------------------------

    print("\n[CHECK 1] Embedding dimension")

    assert tuple(embeddings_1.shape) == (32, 64)

    print("PASS - Query embeddings have shape (32, 64).")

    # --------------------------------------------------------
    # CHECK 2: REPEATABILITY
    # --------------------------------------------------------

    print("\n[CHECK 2] Embedding repeatability")

    embeddings_2 = model(
        inputs,
        training=False
    )

    difference = tf.reduce_max(
        tf.abs(
            embeddings_1 - embeddings_2
        )
    )

    max_difference = float(
        difference.numpy()
    )

    print(
        "Maximum difference between repeated outputs:",
        max_difference
    )

    assert max_difference < 1e-6

    print(
        "PASS - Same input produces the same embedding."
    )

    # --------------------------------------------------------
    # CHECK 3: DIFFERENT USERS
    # --------------------------------------------------------

    print("\n[CHECK 3] Different user representations")

    embedding_difference = tf.reduce_sum(
        tf.abs(
            embeddings_1[0] - embeddings_1[1]
        )
    )

    user_difference = float(
        embedding_difference.numpy()
    )

    print(
        "Difference between first two embeddings:",
        user_difference
    )

    assert user_difference > 0

    print(
        "PASS - Different inputs produce different "
        "representations."
    )

    # --------------------------------------------------------
    # CHECK 4: FINITE VALUES
    # --------------------------------------------------------

    print("\n[CHECK 4] Numerical stability")

    has_nan = bool(
        tf.reduce_any(
            tf.math.is_nan(embeddings_1)
        ).numpy()
    )

    has_inf = bool(
        tf.reduce_any(
            tf.math.is_inf(embeddings_1)
        ).numpy()
    )

    print("Contains NaN:", has_nan)
    print("Contains Inf:", has_inf)

    assert not has_nan
    assert not has_inf

    print(
        "PASS - Embeddings contain only finite values."
    )

    # --------------------------------------------------------
    # CHECK 5: EMBEDDING NORMS
    # --------------------------------------------------------

    print("\n[CHECK 5] Embedding norms")

    norms = tf.norm(
        embeddings_1,
        axis=1
    )

    min_norm = float(
        tf.reduce_min(norms).numpy()
    )

    max_norm = float(
        tf.reduce_max(norms).numpy()
    )

    avg_norm = float(
        tf.reduce_mean(norms).numpy()
    )

    print("Minimum norm:", min_norm)
    print("Maximum norm:", max_norm)
    print("Average norm:", avg_norm)

    assert min_norm > 0

    print(
        "PASS - All query embeddings have non-zero norms."
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL QUERY EMBEDDING QUALITY CHECKS PASSED")
    print("=" * 60)

    print("\nQuery Tower embeddings are:")
    print("- 64-dimensional")
    print("- Repeatable")
    print("- Distinguishable for different inputs")
    print("- Free from NaN and Inf")
    print("- Non-zero and numerically stable")

    print(
        "\nQUERY EMBEDDING QUALITY TEST "
        "COMPLETED SUCCESSFULLY"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_query_embedding_quality()