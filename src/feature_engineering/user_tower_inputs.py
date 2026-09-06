from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "final_training_data.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "query_tower_inputs.csv"


USER_CATEGORICAL_FEATURES = [
    "customer_id_idx",
    "club_member_status_idx",
    "fashion_news_frequency_idx",
    "postal_code_idx",
]


USER_NUMERICAL_FEATURES = [
    "FN",
    "Active",
    "days_since_last_purchase",
    "days_since_first_purchase",
    "avg_inter_purchase_days",
    "purchase_sequence",
]


CONTEXT_FEATURES = [
    "day_of_week",
    "is_weekend",
    "month",
    "quarter",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]


def prepare_query_tower_inputs(df):
    """
    Prepare user and context features for the Query/User Tower.
    """

    required_features = (
        USER_CATEGORICAL_FEATURES
        + USER_NUMERICAL_FEATURES
        + CONTEXT_FEATURES
    )

    missing_features = [
        feature for feature in required_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required Query Tower features: {missing_features}"
        )

    query_df = df[required_features].copy()

    # Fill numerical missing values
    numerical_columns = USER_NUMERICAL_FEATURES + CONTEXT_FEATURES

    for column in numerical_columns:
        query_df[column] = pd.to_numeric(
            query_df[column], errors="coerce"
        )

    query_df[numerical_columns] = query_df[numerical_columns].fillna(0)

    # Convert categorical features to integer indices
    for column in USER_CATEGORICAL_FEATURES:
        query_df[column] = pd.to_numeric(
            query_df[column], errors="coerce"
        ).fillna(0).astype("int32")

    return query_df


def main():
    print("=" * 60)
    print("QUERY / USER TOWER INPUT PREPARATION")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"\nLoading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original dataset shape: {df.shape}")

    query_df = prepare_query_tower_inputs(df)

    print(f"Query Tower input shape: {query_df.shape}")

    print("\nCategorical features:")
    for feature in USER_CATEGORICAL_FEATURES:
        print(
            f"  {feature}: "
            f"min={query_df[feature].min()}, "
            f"max={query_df[feature].max()}"
        )

    print("\nNumerical/context features:")
    for feature in USER_NUMERICAL_FEATURES + CONTEXT_FEATURES:
        print(
            f"  {feature}: "
            f"dtype={query_df[feature].dtype}"
        )

    missing_values = query_df.isnull().sum().sum()

    print(f"\nRemaining missing values: {missing_values}")

    if missing_values != 0:
        raise ValueError("Query Tower input contains missing values.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    query_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved Query Tower inputs to:")
    print(OUTPUT_FILE)

    print("\nFirst 5 rows:")
    print(query_df.head())

    print("\nQUERY / USER TOWER INPUT PREPARATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()