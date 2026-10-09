from pathlib import Path

import pandas as pd

from src.feature_store.redis_store import RedisUserProfileStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_training_data.csv"
)


PROFILE_FEATURES = [
    "customer_id_idx",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "days_since_last_purchase",
    "purchase_sequence",
]


def main():
    print("=" * 70)
    print("REDIS USER PROFILE POPULATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"\nLoading: {INPUT_FILE}")

    required_columns = [
        "t_dat",
        "customer_id",
        *PROFILE_FEATURES,
    ]

    df = pd.read_csv(
        INPUT_FILE,
        usecols=required_columns,
    )

    print(f"Transaction rows loaded: {len(df):,}")

    # Convert transaction date to datetime.
    df["t_dat"] = pd.to_datetime(
        df["t_dat"],
        errors="coerce",
    )

    if df["t_dat"].isna().any():
        raise ValueError(
            "Invalid dates found in t_dat."
        )

    # Sort by customer and date so the last row
    # represents the latest available user context.
    df = df.sort_values(
        ["customer_id_idx", "t_dat"]
    )

    # Keep the latest context row for each user.
    profiles = (
        df.groupby(
            "customer_id_idx",
            as_index=False,
            sort=False,
        )
        .tail(1)
        .copy()
    )

    print(
        f"Unique user profiles to store: "
        f"{len(profiles):,}"
    )

    if len(profiles) != df["customer_id_idx"].nunique():
        raise ValueError(
            "Profile count does not match unique users."
        )

    # Validate QueryTower feature columns.
    missing = [
        column
        for column in PROFILE_FEATURES
        if column not in profiles.columns
    ]

    if missing:
        raise ValueError(
            f"Missing profile features: {missing}"
        )

    # Validate customer ID range.
    min_id = int(profiles["customer_id_idx"].min())
    max_id = int(profiles["customer_id_idx"].max())

    print(f"Customer ID range: {min_id} - {max_id}")

    if min_id < 1 or max_id > 51527:
        raise ValueError(
            f"customer_id_idx out of expected range: "
            f"{min_id} - {max_id}"
        )

    print("\nConnecting to Redis...")

    store = RedisUserProfileStore()

    try:
        if not store.ping():
            raise RuntimeError(
                "Redis ping failed."
            )

        print("Redis connection: OK")

        stored = 0

        for row in profiles.itertuples(index=False):
            profile = {
                "customer_id_idx": int(
                    row.customer_id_idx
                ),
                "month_sin": float(row.month_sin),
                "month_cos": float(row.month_cos),
                "day_of_week_sin": float(
                    row.day_of_week_sin
                ),
                "day_of_week_cos": float(
                    row.day_of_week_cos
                ),
                "is_weekend": int(row.is_weekend),
                "days_since_last_purchase": float(
                    row.days_since_last_purchase
                ),
                "purchase_sequence": float(
                    row.purchase_sequence
                ),
            }

            store.store_user_profile(
                customer_id=int(
                    row.customer_id_idx
                ),
                profile=profile,
            )

            stored += 1

            if stored % 5000 == 0:
                print(
                    f"Stored {stored:,} / "
                    f"{len(profiles):,} profiles"
                )

        print("\n" + "=" * 70)
        print("REDIS USER PROFILE POPULATION COMPLETED")
        print("=" * 70)
        print(f"Profiles stored: {stored:,}")
        print(f"Redis keys created: user:<customer_id_idx>")

    finally:
        store.close()


if __name__ == "__main__":
    main()