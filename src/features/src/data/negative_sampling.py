import numpy as np
import pandas as pd


def generate_negative_samples(
    df: pd.DataFrame,
    num_negatives: int = 4,
    strategy: str = "uniform",
    seed: int = 42,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """
    Generate negative samples for an implicit-feedback dataset.

    Positive interactions receive label=1.
    Sampled unobserved interactions receive label=0.

    Args:
        df: DataFrame containing user and item interaction columns.
        num_negatives: Number of negatives per unique positive item.
        strategy: "uniform" or "popularity".
        seed: Random seed.
        user_col: User identifier column.
        item_col: Item identifier column.

    Returns:
        DataFrame containing user, item and label columns.
    """

    if df.empty:
        return pd.DataFrame(
            columns=[user_col, item_col, "label"]
        )

    if num_negatives < 0:
        raise ValueError("num_negatives must be >= 0")

    if strategy not in {"uniform", "popularity"}:
        raise ValueError(
            "strategy must be either 'uniform' or 'popularity'"
        )

    required_columns = {user_col, item_col}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    rng = np.random.default_rng(seed)

    # All available items
    all_items = np.asarray(df[item_col].dropna().unique())

    if len(all_items) == 0:
        raise ValueError("No items available for negative sampling.")

    # Positive interactions per user
    user_positives = (
        df.groupby(user_col)[item_col]
        .apply(set)
        .to_dict()
    )

    # Popularity-weighted probabilities
    if strategy == "popularity":
        item_counts = df[item_col].value_counts()

        weights = np.array(
            [
                float(item_counts.get(item, 1)) ** 0.75
                for item in all_items
            ],
            dtype=np.float64,
        )

        probabilities = weights / weights.sum()
    else:
        probabilities = None

    # Positive samples
    positives = df[[user_col, item_col]].drop_duplicates().copy()
    positives["label"] = 1

    negative_rows = []

    for user, positive_items in user_positives.items():

        if num_negatives == 0:
            continue

        available_items = np.array(
            [item for item in all_items if item not in positive_items]
        )

        # User has interacted with every available item.
        if len(available_items) == 0:
            continue

        required = len(positive_items) * num_negatives

        # Prefer sampling without replacement when possible.
        if required <= len(available_items):

            if strategy == "popularity":
                available_weights = np.array(
                    [
                        float(
                            df[item_col].value_counts().get(item, 1)
                        ) ** 0.75
                        for item in available_items
                    ],
                    dtype=np.float64,
                )
                available_probs = (
                    available_weights / available_weights.sum()
                )
            else:
                available_probs = None

            sampled_items = rng.choice(
                available_items,
                size=required,
                replace=False,
                p=available_probs,
            )

        else:
            # If more negatives are required than unique unseen items,
            # sample with replacement.
            if strategy == "popularity":
                available_weights = np.array(
                    [
                        float(
                            df[item_col].value_counts().get(item, 1)
                        ) ** 0.75
                        for item in available_items
                    ],
                    dtype=np.float64,
                )
                available_probs = (
                    available_weights / available_weights.sum()
                )
            else:
                available_probs = None

            sampled_items = rng.choice(
                available_items,
                size=required,
                replace=True,
                p=available_probs,
            )

        for item in sampled_items:
            negative_rows.append(
                {
                    user_col: user,
                    item_col: item,
                    "label": 0,
                }
            )

    negatives = pd.DataFrame(
        negative_rows,
        columns=[user_col, item_col, "label"],
    )

    result = pd.concat(
        [positives, negatives],
        ignore_index=True,
    )

    result = (
        result
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    return result


if __name__ == "__main__":

    raw_data = {
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [101, 102, 102, 103, 101],
    }

    df_interactions = pd.DataFrame(raw_data)

    print("=" * 60)
    print("MEMBER 4 - NEGATIVE SAMPLING VERIFICATION")
    print("=" * 60)

    training_data = generate_negative_samples(
        df_interactions,
        num_negatives=2,
        strategy="uniform",
        seed=42,
    )

    print("\nOriginal interactions:")
    print(df_interactions)

    print("\nGenerated training data:")
    print(training_data)

    print("\nClass distribution:")
    print(training_data["label"].value_counts())

    print("\nTotal samples:", len(training_data))
    print("Positive samples:", (training_data["label"] == 1).sum())
    print("Negative samples:", (training_data["label"] == 0).sum())

    print("\nNEGATIVE SAMPLING VERIFICATION PASSED")