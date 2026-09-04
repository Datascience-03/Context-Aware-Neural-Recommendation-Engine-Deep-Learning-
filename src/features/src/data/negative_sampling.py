import numpy as np
import pandas as pd


def generate_negative_samples(
    df: pd.DataFrame,
    num_negatives: int = 4,
    strategy: str = "uniform",  # Options: 'uniform' or 'popularity'
    seed: int = 42,
) -> pd.DataFrame:
    """Generates negative samples for implicit feedback datasets.

    Args:
        df: DataFrame containing at least ['user_id', 'item_id'].
        num_negatives: Number of negative samples to draw per positive
          interaction.
        strategy: 'uniform' for random sampling, 'popularity' for
          popularity-weighted.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Formatted dataset with columns ['user_id', 'item_id',
        'label'].
    """
    np.random.seed(seed)

    # 1. Build set of all unique items
    all_items = np.array(df["item_id"].unique())
    num_items = len(all_items)

    # 2. Task 1: Find observed interactions per user (Candidate pool logic)
    # Negatives are sampled only from unobserved items (all_items - user_positives)
    user_positives = df.groupby("user_id")["item_id"].apply(set).to_dict()

    # Precompute item probabilities for popularity-weighted sampling
    if strategy == "popularity":
        item_counts = df["item_id"].value_counts()
        # Common practice in recommendation/Word2Vec: smooth frequencies with power 0.75
        item_weights = np.array(
            [item_counts.get(item, 1) ** 0.75 for item in all_items]
        )
        sampling_probs = item_weights / item_weights.sum()
    else:
        sampling_probs = None

    # Prepare positive rows: label = 1
    positives = df[["user_id", "item_id"]].copy()
    positives["label"] = 1

    # 3. Task 2: Implement negative sampling logic
    neg_users = []
    neg_items = []

    for user, pos_set in user_positives.items():
        # Number of negatives needed for this user
        needed = len(pos_set) * num_negatives

        sampled_count = 0
        while sampled_count < needed:
            # Oversample in batches for performance
            batch_size = max(needed - sampled_count, 100)
            candidates = np.random.choice(
                all_items, size=batch_size, replace=True, p=sampling_probs
            )

            # Filter out observed (positive) items
            valid_negatives = [
                item for item in candidates if item not in pos_set
            ]

            take = min(len(valid_negatives), needed - sampled_count)
            neg_items.extend(valid_negatives[:take])
            neg_users.extend([user] * take)
            sampled_count += take

    # 4. Task 3: Format paired data with label 0 for negatives
    negatives = pd.DataFrame(
        {"user_id": neg_users, "item_id": neg_items, "label": 0}
    )

    # Combine positives and negatives and shuffle
    final_df = pd.concat([positives, negatives], ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=seed).reset_index(
        drop=True
    )

    return final_df


# ==========================================
# Quick Verification / Test Run
# ==========================================
if __name__ == "__main__":
    # Dummy interaction data (Day 1 output simulation)
    raw_data = {
        "user_id": [1, 1, 2, 2, 3],
        "item_id": [101, 102, 102, 103, 101],
    }
    df_interactions = pd.DataFrame(raw_data)

    print("--- Original Positive Interactions ---")
    print(df_interactions)

    # Run sampling with 2 negatives per positive
    training_data = generate_negative_samples(
        df_interactions, num_negatives=2, strategy="uniform"
    )

    print("\n--- Formatted Training Data (Positives + Negatives) ---")
    print(training_data)
    print("\nClass distribution:")
    print(training_data["label"].value_counts())