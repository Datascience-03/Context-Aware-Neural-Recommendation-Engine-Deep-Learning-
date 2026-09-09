import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def export_features(
    df: pd.DataFrame,
    output_path: str,
    compression: str = "snappy",
    index: bool = False
) -> str:
    """
    Exports the processed contextual feature table to Parquet format.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Cannot export an empty DataFrame.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_rows, num_cols = df.shape
    logging.info(f"Feature Table Shape: {num_rows} rows x {num_cols} columns")

    try:
        df.to_parquet(
            output_path,
            engine="pyarrow",
            compression=compression,
            index=index
        )
        logging.info(f"Successfully exported features to: {output_path}")
        return os.path.abspath(output_path)

    except Exception as e:
        logging.error(f"Failed to export feature table: {e}")
        raise


# --- Main Execution Block (for testing) ---
if __name__ == "__main__":
    logging.info("Generating sample contextual feature data...")

    # 1. Simulate data coming from Member 2 & Member 5
    sample_data = {
        "entity_id": [101, 102, 103, 104, 105],
        "timestamp": pd.date_range(start="2025-01-01", periods=5, freq="h"),
        "device_context": [0, 1, 0, 2, 1],
        "time_of_day": [0, 1, 2, 3, 4],
        "day_of_week": [2, 2, 2, 2, 2],
        "session_duration": [120.5, 45.0, 300.2, 15.8, 90.0],
        "rolling_interaction_count": [1, 3, 5, 2, 4],
    }
    df = pd.DataFrame(sample_data)

    # 2. Define the output path
    output_file = "data/processed/contextual_features.parquet"

    # 3. Run export_features
    exported_path = export_features(df, output_path=output_file)

    # 4. Quick verification: Read the parquet file back
    read_df = pd.read_parquet(exported_path)
    print("\n--- Verified Parquet File Content ---")
    print(read_df.head())
    print("\nFile shape:", read_df.shape)