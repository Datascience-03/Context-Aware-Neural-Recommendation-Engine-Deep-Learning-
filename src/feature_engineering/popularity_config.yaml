import pandas as pd
import numpy as np

def generate_rolling_popularity_features(
    df: pd.DataFrame, 
    date_col: str = 't_dat', 
    item_col: str = 'article_id',
    windows: list = [7, 14, 30]
) -> pd.DataFrame:
    """
    Computes rolling sales counts for each item over specified day windows.
    Fills zero-sales windows with 0.
    
    Parameters:
    - df: Input transaction DataFrame.
    - date_col: Name of the date column.
    - item_col: Name of the item identifier column.
    - windows: List of day windows to calculate rolling counts for.
    
    Returns:
    - DataFrame containing [date, article_id] and rolling popularity counts.
    """
    # 1. Ensure date is in datetime format
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 2. Aggregate daily sales per item
    daily_sales = (
        df.groupby([item_col, date_col])
        .size()
        .reset_index(name='daily_count')
    )
    
    # 3. Create a complete grid of (all_items x all_dates) to account for 0-sales days
    unique_items = df[item_col].unique()
    all_dates = pd.date_range(
        start=df[date_col].min(), 
        end=df[date_col].max(), 
        freq='D'
    )
    
    full_index = pd.MultiIndex.from_product(
        [unique_items, all_dates], 
        names=[item_col, date_col]
    )
    
    # 4. Reindex to include missing days with 0 sales
    daily_sales_full = (
        daily_sales.set_index([item_col, date_col])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    
    # 5. Sort before applying rolling windows
    daily_sales_full = daily_sales_full.sort_values(by=[item_col, date_col])
    
    # 6. Calculate rolling sales counts for each window (7d, 14d, 30d)
    for w in windows:
        col_name = f'sales_last_{w}d'
        daily_sales_full[col_name] = (
            daily_sales_full
            .groupby(item_col)['daily_count']
            .transform(lambda s: s.rolling(window=w, min_periods=1).sum())
            .astype(np.int32)
        )
    
    return daily_sales_full


# ==========================================
# Example Usage & Verification
# ==========================================
if __name__ == '__main__':
    # Sample synthetic transaction data
    sample_data = {
        't_dat': [
            '2023-01-01', '2023-01-01', '2023-01-02', 
            '2023-01-05', '2023-01-10', '2023-01-20',
            '2023-01-01', '2023-01-15'
        ],
        'article_id': [
            'A101', 'A101', 'A101', 
            'A101', 'A101', 'A101',
            'B202', 'B202'
        ]
    }
    
    df_transactions = pd.DataFrame(sample_data)
    
    # Compute rolling features
    popularity_df = generate_rolling_popularity_features(
        df=df_transactions,
        date_col='t_dat',
        item_col='article_id',
        windows=[7, 14, 30]
    )
    
    print("--- Sample Output ---")
    print(popularity_df.head(15))