import numpy as np
import pandas as pd


def generate_recency_frequency_features(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "transaction_date",
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
    # 1. Ensure datetime format and sort chronologically
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[customer_col, date_col]).reset_index(drop=True)

    # 2. Cumulative purchase count (1st purchase = 1, 2nd = 2, ...)
    df["purchase_sequence"] = df.groupby(customer_col).cumcount() + 1

    # 3. Days since previous purchase (Recency relative to current transaction)
    # Use shift(1) per customer to look at the previous transaction
    df["prev_transaction_date"] = df.groupby(customer_col)[date_col].shift(1)
    df["days_since_last_purchase"] = (
        df[date_col] - df["prev_transaction_date"]
    ).dt.total_seconds() / (24 * 3600)

    # 4. Days since first purchase (Customer Tenure)
    first_purchase_dates = df.groupby(customer_col)[date_col].transform("first")
    df["days_since_first_purchase"] = (
        df[date_col] - first_purchase_dates
    ).dt.total_seconds() / (24 * 3600)

    # 5. Average inter-purchase duration (Historical average gap up to this point)
    # Expanding mean of the inter-purchase gaps per customer
    df["avg_inter_purchase_days"] = df.groupby(customer_col)[
        "days_since_last_purchase"
    ].transform(lambda x: x.expanding().mean())

    # 6. Fallbacks for first-time buyers (NaNs)
    # For a user's 1st purchase, 'days_since_last_purchase' and 'avg_inter_purchase_days' are NaN
    df["is_first_purchase"] = (
        df["days_since_last_purchase"].isna().astype(int)
    )  # indicator flag

    df["days_since_last_purchase"] = df["days_since_last_purchase"].fillna(
        fallback_value
    )
    df["avg_inter_purchase_days"] = df["avg_inter_purchase_days"].fillna(
        fallback_value
    )

    # Clean up temporary column
    df = df.drop(columns=["prev_transaction_date"])

    return df


# -------------------------------------------------------------
# Example Usage / Test
# -------------------------------------------------------------
if __name__ == "__main__":
    # Sample transaction data
    sample_data = {
        "customer_id": [101, 101, 101, 102, 102],
        "transaction_date": [
            "2023-01-01",
            "2023-01-10",
            "2023-01-25",
            "2023-02-01",
            "2023-02-05",
        ],
        "amount": [50.0, 30.0, 100.0, 20.0, 45.0],
    }

    df_sample = pd.DataFrame(sample_data)
    df_featured = generate_recency_frequency_features(df_sample)

    print(
        df_featured[
            [
                "customer_id",
                "transaction_date",
                "purchase_sequence",
                "days_since_last_purchase",
                "days_since_first_purchase",
                "avg_inter_purchase_days",
                "is_first_purchase",
            ]
        ]
    )