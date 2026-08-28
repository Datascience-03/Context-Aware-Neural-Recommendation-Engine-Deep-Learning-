import numpy as np
import pandas as pd


def get_season(month: int) -> str:
    """Map month to season."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"


def extract_calendar_features(df: pd.DataFrame, timestamp_col: str = "t_dat") -> pd.DataFrame:
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


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add sine and cosine cyclical transformations for periodic features."""
    df = df.copy()

    # Cyclical encoding for day_of_week (period = 7)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Cyclical encoding for month (period = 12)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def generate_recency_frequency_features(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "t_dat",
    fallback_value: float = -1.0,
) -> pd.DataFrame:
    """Computes user recency, tenure, and purchase frequency features.

    Parameters:
    -----------
    df : pd.DataFrame
        The transactions dataframe.
    customer_col : str
        Column name for customer/user identifier.
    date_col : str
        Column name for transaction timestamp or date.
    fallback_value : float
        Value to impute for first-time buyers (default: -1.0).

    Returns:
    --------
    pd.DataFrame with new recency and frequency feature columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Store original order index
    df['__orig_idx'] = range(len(df))
    df = df.sort_values(by=[customer_col, date_col]).reset_index(drop=True)

    # Cumulative purchase count (1st purchase = 1, 2nd = 2, ...)
    df["purchase_sequence"] = df.groupby(customer_col).cumcount() + 1

    # Days since previous purchase (Recency relative to current transaction)
    df["prev_transaction_date"] = df.groupby(customer_col)[date_col].shift(1)
    df["days_since_last_purchase"] = (
        df[date_col] - df["prev_transaction_date"]
    ).dt.total_seconds() / (24 * 3600)

    # Days since first purchase (Customer Tenure)
    first_purchase_dates = df.groupby(customer_col)[date_col].transform("first")
    df["days_since_first_purchase"] = (
        df[date_col] - first_purchase_dates
    ).dt.total_seconds() / (24 * 3600)

    # Average inter-purchase duration (Historical average gap up to this point)
    df["avg_inter_purchase_days"] = df.groupby(customer_col)[
        "days_since_last_purchase"
    ].transform(lambda x: x.expanding().mean())

    # Fallbacks for first-time buyers (NaNs)
    df["is_first_purchase"] = (
        df["days_since_last_purchase"].isna().astype(int)
    )

    df["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(
        fallback_value
    )
    df["avg_inter_purchase_days"] = df["avg_inter_purchase_days"].fillna(
        fallback_value
    )

    df = df.drop(columns=["prev_transaction_date"])
    
    # Restore original sorting
    df = df.sort_values(by='__orig_idx').drop(columns=['__orig_idx']).reset_index(drop=True)
    return df


def compute_product_popularity_over_time(
    df: pd.DataFrame,
    item_col: str = "article_id",
    date_col: str = "t_dat"
) -> pd.DataFrame:
    """
    Computes product popularity over time (rolling 30-day purchase count for each item).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort chronologically to apply rolling correctly
    df = df.sort_values(by=date_col)
    
    # Count daily sales for each item
    daily_sales = df.groupby([date_col, item_col]).size().reset_index(name='daily_count')
    daily_sales = daily_sales.set_index(date_col).sort_index()
    
    popularity_list = []
    # Group by article and compute rolling 30d sum
    for item_id, group in daily_sales.groupby(item_col):
        group = group.copy()
        # rolling 30-day sum of daily_count
        group['popularity_over_time'] = group['daily_count'].rolling('30D', min_periods=1).sum()
        popularity_list.append(group)
        
    popularity_df = pd.concat(popularity_list).reset_index()
    
    # Merge back into original dataframe
    df = df.merge(
        popularity_df[[date_col, item_col, 'popularity_over_time']],
        on=[date_col, item_col],
        how='left'
    )
    
    df['popularity_over_time'] = df['popularity_over_time'].fillna(1.0)
    return df


# -------------------------------------------------------------
# Example Usage / Test
# -------------------------------------------------------------
if __name__ == "__main__":
    # Test transactions
    tx_df = pd.DataFrame({
        'customer_id': ['U1', 'U1', 'U1', 'U2', 'U2'],
        't_dat': ['2023-01-01', '2023-01-10', '2023-01-25', '2023-02-01', '2023-02-05'],
        'article_id': [1001, 1002, 1001, 1003, 1001]
    })
    
    print("Testing extraction...")
    tx_df = extract_calendar_features(tx_df)
    tx_df = add_cyclical_encoding(tx_df)
    tx_df = generate_recency_frequency_features(tx_df)
    tx_df = compute_product_popularity_over_time(tx_df)
    print(tx_df)