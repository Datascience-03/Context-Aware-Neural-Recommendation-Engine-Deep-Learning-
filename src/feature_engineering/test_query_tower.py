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
# LOAD REAL QUERY TOWER INPUTS
# ============================================================

def load_real_query_inputs(sample_size=32):
    """
    Load real Query Tower input features from the processed dataset.

    Categorical features are stored in the CSV using *_idx columns.
    Numerical/contextual features use their original feature names.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Query Tower input file not found: {INPUT_FILE}"
        )

    # Load a small real-data batch for validation
    df = pd.read_csv(
        INPUT_FILE,
        nrows=sample_size
    )

    inputs = {}

    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    for feature in USER_CATEGORICAL_FEATURES:

        # query_tower_inputs.csv stores categorical
        # features using their vocabulary index columns.
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
                f"Required numerical feature '{feature}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        inputs[feature] = tf.convert_to_tensor(
            df[feature].values,
            dtype=tf.float32
        )

    return df, inputs


# ============================================================
# TEST QUERY TOWER WITH REAL DATA
# ============================================================

def test_real_query_tower():

    print("=" * 60)
    print("REAL DATA QUERY TOWER TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: LOAD REAL DATA
    # --------------------------------------------------------

    print("\nLoading real Query Tower data...")

    df, inputs = load_real_query_inputs(
        sample_size=32
    )

    print(f"Loaded test batch: {len(df)} rows")

    # --------------------------------------------------------
    # STEP 2: LOAD EMBEDDING CONFIGURATION
    # --------------------------------------------------------

    print("\nLoading user embedding configuration...")

    embedding_config = load_embedding_config()

    print("Embedding configuration loaded successfully.")

    # --------------------------------------------------------
    # STEP 3: CREATE QUERY TOWER
    # --------------------------------------------------------

    print("\nCreating Query Tower...")

    model = QueryTower(
        embedding_config=embedding_config,
        output_dim=64
    )

    print("Query Tower created successfully.")

    # --------------------------------------------------------
    # STEP 4: RUN REAL DATA THROUGH QUERY TOWER
    # --------------------------------------------------------

    print("\nRunning Query Tower...")

    query_embeddings = model(
        inputs,
        training=False
    )

    print(
        "Query embedding shape:",
        query_embeddings.shape
    )

    # --------------------------------------------------------
    # STEP 5: VALIDATE OUTPUT SHAPE
    # --------------------------------------------------------

    expected_shape = (32, 64)

    print(
        "Expected shape:",
        expected_shape
    )

    assert tuple(query_embeddings.shape) == expected_shape, (
        f"Unexpected query embedding shape: "
        f"{query_embeddings.shape}"
    )

    print("Output shape validation passed.")

    # --------------------------------------------------------
    # STEP 6: CHECK NaN VALUES
    # --------------------------------------------------------

    has_nan = bool(
        tf.reduce_any(
            tf.math.is_nan(query_embeddings)
        ).numpy()
    )

    print(
        "Contains NaN:",
        has_nan
    )

    assert not has_nan, (
        "Query embeddings contain NaN values."
    )

    # --------------------------------------------------------
    # STEP 7: CHECK INFINITE VALUES
    # --------------------------------------------------------

    has_inf = bool(
        tf.reduce_any(
            tf.math.is_inf(query_embeddings)
        ).numpy()
    )

    print(
        "Contains Inf:",
        has_inf
    )

    assert not has_inf, (
        "Query embeddings contain infinite values."
    )

    # --------------------------------------------------------
    # STEP 8: CHECK EMBEDDING NORMS
    # --------------------------------------------------------

    norms = tf.norm(
        query_embeddings,
        axis=1
    )

    min_norm = float(
        tf.reduce_min(norms).numpy()
    )

    max_norm = float(
        tf.reduce_max(norms).numpy()
    )

    print(
        "\nEmbedding norm range:",
        min_norm,
        "to",
        max_norm
    )

    # Every embedding should contain meaningful values
    assert bool(
        tf.reduce_all(
            norms > 0
        ).numpy()
    ), "One or more query embeddings have zero norm."

    print("Embedding norm validation passed.")

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\nReal data Query Tower test passed.")

    print(
        "\nREAL DATA QUERY TOWER TEST COMPLETED SUCCESSFULLY"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_real_query_tower()