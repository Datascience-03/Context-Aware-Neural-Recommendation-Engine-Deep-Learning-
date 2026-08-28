import os
import sys
import pandas as pd

# Append workspace root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data_processing.missing_values import handle_missing_values
from src.data_processing.cold_start import handle_user_cold_start, handle_item_cold_start
from src.features.contextual_features import (
    extract_calendar_features,
    add_cyclical_encoding,
    generate_recency_frequency_features,
    compute_product_popularity_over_time
)
from src.feature_engineering.vocabularies import FeatureVocabularies


def run_integration_pipeline():
    print("=== STARTING INTEGRATION & VALIDATION PIPELINE ===")
    
    base_path = "data/reduced"
    processed_path = "data/processed"
    
    # 1. Check if raw data exists, if not generate it
    if not os.path.exists(f"{base_path}/transactions_reduced.csv"):
        print("Reduced dataset not found. Running generate_data.py...")
        import subprocess
        subprocess.run(["python", "generate_data.py"], check=True)

    # 2. Run Member 2 pipeline to clean and process data
    print("Step 1: Running PySpark/Pandas Data Processing Pipeline...")
    from src.pipeline.pyspark_pipeline import main as run_pyspark_pipeline
    run_pyspark_pipeline()
    
    # 3. Load processed data
    print("Step 2: Loading cleaned datasets...")
    transactions = pd.read_parquet(f"{processed_path}/transactions_processed.parquet")
    customers = pd.read_parquet(f"{processed_path}/customers_processed.parquet")
    articles = pd.read_parquet(f"{processed_path}/articles_processed.parquet")
    
    print(f"Loaded Cleaned Data: {len(transactions)} transactions, {len(customers)} customers, {len(articles)} articles.")

    # 4. Handle Missing Values - Member 3 Task
    print("Step 3: Merging transactions and articles for missing value handling...")
    # Merge transactions and articles
    tx_articles = transactions.merge(articles, on="article_id", how="left")
    
    # Run missing value handler
    tx_articles = handle_missing_values(tx_articles)
    
    # 5. Extract Contextual Features - Member 4 Task
    print("Step 4: Extracting contextual and time-based features...")
    # Time and Calendar features
    tx_articles = extract_calendar_features(tx_articles, timestamp_col="t_dat")
    tx_articles = add_cyclical_encoding(tx_articles)
    
    # Recency features
    tx_articles = generate_recency_frequency_features(tx_articles, customer_col="customer_id", date_col="t_dat")
    
    # Product popularity over time
    tx_articles = compute_product_popularity_over_time(tx_articles, item_col="article_id", date_col="t_dat")
    
    # 6. Fit Vocabularies - Member 5 Task
    print("Step 5: Fitting feature vocabularies for embedding lookups...")
    vocab_builder = FeatureVocabularies()
    vocab_builder.fit_all(customers, articles)
    vocab_builder.save(f"{processed_path}/vocabularies.json")
    
    # 7. Apply Vocabularies and Transform Categoricals
    print("Step 6: Transforming categorical features into integer indices...")
    # Join with customer info to have all features in the final table
    final_df = tx_articles.merge(customers, on="customer_id", how="left")
    
    # Fill remaining NaNs in customer columns
    final_df['FN'] = final_df['FN'].fillna(0.0)
    final_df['Active'] = final_df['Active'].fillna(0.0)
    final_df['club_member_status'] = final_df['club_member_status'].fillna("Unknown")
    final_df['fashion_news_frequency'] = final_df['fashion_news_frequency'].fillna("Unknown")
    
    categorical_columns = [
        "customer_id", "club_member_status", "fashion_news_frequency", "postal_code",
        "article_id", "prod_name", "product_type_name", "product_group_name",
        "graphical_appearance_name", "colour_group_name", "department_name", "index_name"
    ]
    
    final_df = vocab_builder.transform(final_df, categorical_columns)
    
    # 8. Save Final Training Dataset
    print("Step 7: Validating and saving final training dataset...")
    
    # Validation checks
    assert not final_df.empty, "Error: Integrated dataset is empty!"
    assert "days_since_last_purchase" in final_df.columns, "Error: Recency features missing!"
    assert "popularity_over_time" in final_df.columns, "Error: Product popularity feature missing!"
    assert "customer_id_idx" in final_df.columns, "Error: Vocabulary transformations missing!"
    assert final_df.isnull().sum().sum() == 0, f"Error: Found null values in final dataset:\n{final_df.isnull().sum()}"
    
    final_output_path = f"{processed_path}/final_training_data.csv"
    final_df.to_csv(final_output_path, index=False)
    print(f"SUCCESS! Integrated final training dataset saved to: {final_output_path}")
    print(f"Final dataset dimensions: {final_df.shape}")
    print("Sample rows:")
    print(final_df[[
        "customer_id_idx", "article_id_idx", "month_sin", "days_since_last_purchase", "popularity_over_time"
    ]].head())


if __name__ == "__main__":
    run_integration_pipeline()
