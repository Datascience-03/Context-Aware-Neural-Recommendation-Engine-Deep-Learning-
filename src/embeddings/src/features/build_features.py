import numpy as np
import pandas as pd


def get_season(month):
    """Map month to season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def extract_calendar_features(df, timestamp_col="timestamp"):
    """Extract basic calendar and seasonal features."""
    df = df.copy()

    # Ensure column is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # 1. Basic Calendar Features
    df["day_of_week"] = (
        df[timestamp_col].dt.dayofweek + 1
    )  # 1 = Monday, 7 = Sunday
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)
    df["month"] = df[timestamp_col].dt.month
    df["quarter"] = df[timestamp_col].dt.quarter
    df["season"] = df["month"].apply(get_season)

    return df


def add_cyclical_encoding(df):
    """Add sine and cosine cyclical transformations for periodic features."""
    df = df.copy()

    # Cyclical encoding for day_of_week (period = 7)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Cyclical encoding for month (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


# --- Example Usage ---
if __name__ == "__main__":
    # Sample data for testing
    data = {
        "transaction_id": [1, 2, 3, 4],
        "timestamp": [
            "2024-01-15 10:30:00",
            "2024-04-20 14:00:00",
            "2024-07-06 18:45:00",
            "2024-10-31 09:15:00",
        ],
    }
    df = pd.DataFrame(data)

    # Apply transformations
    df = extract_calendar_features(df, timestamp_col="timestamp")
    df = add_cyclical_encoding(df)

    print(df.head())