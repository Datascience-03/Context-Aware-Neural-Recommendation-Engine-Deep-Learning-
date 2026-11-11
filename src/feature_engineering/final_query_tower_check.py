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
# LOAD QUERY TOWER INPUTS
# ============================================================

def load_query_inputs(sample_size=32):

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
                f"Missing categorical column: {column_name}"
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
                f"Missing numerical feature: {feature}"
            )

        inputs[feature] = tf.convert_to_tensor(
            df[feature].values,
            dtype=tf.float32
        )

    return df, inputs


# ============================================================
# FINAL QUERY TOWER CHECK
# ============================================================

def run_final_check():

    print("=" * 65)
    print("FINAL QUERY TOWER INTEGRATION CHECK")
    print("=" * 65)

    # --------------------------------------------------------
    # CHECK 1: INPUT DATA
    # --------------------------------------------------------

    print("\n[CHECK 1] Query Tower input data")

    df, inputs = load_query_inputs(
        sample_size=32
    )

    print("Loaded records:", len(df))
    print("Input feature count:", len(inputs))

    assert len(df) == 32

    print("PASS - Real Query Tower input data loaded.")

    # --------------------------------------------------------
    # CHECK 2: EMBEDDING CONFIGURATION
    # --------------------------------------------------------

    print("\n[CHECK 2] Embedding configuration")

    embedding_config = load_embedding_config()

    print(
        "Configured categorical features:",
        list(embedding_config.keys())
    )

    for feature in USER_CATEGORICAL_FEATURES:

        assert feature in embedding_config

        print(
            f"{feature}:",
            embedding_config[feature]
        )

    print("PASS - Embedding configuration loaded.")

    # --------------------------------------------------------
    # CHECK 3: QUERY TOWER CREATION
    # --------------------------------------------------------

    print("\n[CHECK 3] Query Tower creation")

    model = QueryTower(
        embedding_config=embedding_config,
        output_dim=64
    )

    print("Query Tower created successfully.")

    # --------------------------------------------------------
    # CHECK 4: FORWARD PASS
    # --------------------------------------------------------

    print("\n[CHECK 4] Query Tower forward pass")

    embeddings = model(
        inputs,
        training=False
    )

    print(
        "Query embedding shape:",
        embeddings.shape
    )

    assert tuple(embeddings.shape) == (32, 64)

    print(
        "PASS - Query Tower generates 64-dimensional embeddings."
    )

    # --------------------------------------------------------
    # CHECK 5: NUMERICAL VALIDITY
    # --------------------------------------------------------

    print("\n[CHECK 5] Numerical validity")

    has_nan = bool(
        tf.reduce_any(
            tf.math.is_nan(embeddings)
        ).numpy()
    )

    has_inf = bool(
        tf.reduce_any(
            tf.math.is_inf(embeddings)
        ).numpy()
    )

    print("Contains NaN:", has_nan)
    print("Contains Inf:", has_inf)

    assert not has_nan
    assert not has_inf

    print(
        "PASS - Embeddings contain valid finite values."
    )

    # --------------------------------------------------------
    # CHECK 6: EMBEDDING NORMS
    # --------------------------------------------------------

    print("\n[CHECK 6] Embedding norms")

    norms = tf.norm(
        embeddings,
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
        "PASS - Query embeddings have non-zero norms."
    )

    # --------------------------------------------------------
    # CHECK 7: REPEATABILITY
    # --------------------------------------------------------

    print("\n[CHECK 7] Query embedding repeatability")

    embeddings_repeat = model(
        inputs,
        training=False
    )

    max_difference = float(
        tf.reduce_max(
            tf.abs(
                embeddings - embeddings_repeat
            )
        ).numpy()
    )

    print(
        "Maximum repeated-output difference:",
        max_difference
    )

    assert max_difference < 1e-6

    print(
        "PASS - Query Tower produces consistent embeddings."
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("ALL FINAL QUERY TOWER CHECKS PASSED")
    print("=" * 65)

    print("\nWeek 2 Member 1 Query Tower is integration-ready.")

    print("\nValidated:")
    print("- Real user input features")
    print("- Embedding configuration")
    print("- Query Tower architecture")
    print("- 64-dimensional query embeddings")
    print("- Numerical stability")
    print("- Non-zero embedding norms")
    print("- Embedding repeatability")

    print(
        "\nFINAL QUERY TOWER CHECK COMPLETED SUCCESSFULLY"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_final_check()