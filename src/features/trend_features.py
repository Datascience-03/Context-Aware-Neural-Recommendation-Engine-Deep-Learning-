import numpy as np
import pandas as pd
from typing import Optional


def compute_popularity_trend_ratio(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    item_col: str = "item_id",
    date_col: str = "transaction_date",
    epsilon: float = 1e-5,
) -> pd.DataFrame:
    """
    Calculates Popularity Momentum/Trend Ratio:
        Trend Ratio = Sales_last_7_days / ((Sales_last_30_days / 4) + epsilon)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Filter to the last 30 days window
    start_30d = reference_date - pd.Timedelta(days=30)
    start_7d = reference_date - pd.Timedelta(days=7)

    mask_30d = (df[date_col] >= start_30d) & (df[date_col] <= reference_date)
    mask_7d = (df[date_col] >= start_7d) & (df[date_col] <= reference_date)

    # Aggregate counts
    sales_30d = (
        df[mask_30d]
        .groupby(item_col)
        .size()
        .rename("sales_30d")
    )
    sales_7d = (
        df[mask_7d]
        .groupby(item_col)
        .size()
        .rename("sales_7d")
    )

    # Combine metrics
    all_items = pd.Series(df[item_col].unique(), name=item_col)
    result = pd.DataFrame(all_items)
    result = result.merge(sales_30d, on=item_col, how="left").fillna(0)
    result = result.merge(sales_7d, on=item_col, how="left").fillna(0)

    # Calculate Trend Ratio
    result["trend_ratio"] = result["sales_7d"] / ((result["sales_30d"] / 4.0) + epsilon)

    return result[[item_col, "sales_7d", "sales_30d", "trend_ratio"]]


def compute_exponential_decay_score(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    item_col: str = "item_id",
    date_col: str = "transaction_date",
    decay_lambda: Optional[float] = None,
    half_life_days: float = 7.0,
) -> pd.DataFrame:
    """
    Calculates Exponential Time-Decayed Popularity:
        DecayScore(item) = sum(exp(-lambda * delta_t))
        where lambda = ln(2) / half_life_days (if not explicitly provided)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if decay_lambda is None:
        decay_lambda = np.log(2) / half_life_days

    # Filter events up to reference date
    valid_df = df[df[date_col] <= reference_date].copy()

    # Calculate delta_t in days
    delta_days = (reference_date - valid_df[date_col]).dt.total_seconds() / (24 * 3600)
    valid_df["decay_weight"] = np.exp(-decay_lambda * delta_days)

    # Aggregate scores
    decay_scores = (
        valid_df.groupby(item_col)["decay_weight"]
        .sum()
        .rename("decay_score")
        .reset_index()
    )

    # Include cold/unseen items with 0 score
    all_items = pd.DataFrame({item_col: df[item_col].unique()})
    result = all_items.merge(decay_scores, on=item_col, how="left").fillna(0.0)

    return result