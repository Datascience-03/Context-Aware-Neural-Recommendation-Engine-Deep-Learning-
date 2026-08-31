import os
import logging
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def export_features(
    df: pd.DataFrame,
    output_path: str,
    compression: str = "snappy",
    index: bool = False
) -> str:
    """
    Exports the processed contextual feature table to Parquet format.

    Validates schema consistency, checks for empty datasets, logs feature
    dimensions/shapes, and handles directory creation before export.

    Parameters
    ----------
    df : pd.DataFrame
        The merged and processed feature DataFrame ready for downstream modeling.
    output_path : str
        The destination file path (e.g., 'data/processed/contextual_features.parquet').
    compression : str, default 'snappy'
        Compression algorithm for Parquet ('snappy', 'gzip', 'brotli', 'none').
    index : bool, default False
        Whether to write the DataFrame index as a column.

    Returns
    -------
    str
        The absolute path to the saved Parquet file.

    Raises
    ------
    ValueError
        If the input DataFrame is empty or not a pandas DataFrame.
    IOError
        If writing to the specified path fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if df.empty:
        raise ValueError("Cannot export an empty DataFrame.")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Log feature metadata and shape
    num_rows, num_cols = df.shape
    logging.info(f"Preparing to export feature table.")
    logging.info(f"Feature Table Shape: {num_rows} rows x {num_cols} columns")
    logging.info(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    try:
        # Export to Parquet
        df.to_parquet(
            output_path,
            engine="pyarrow",  # or 'fastparquet'
            compression=compression,
            index=index
        )
        logging.info(f"Successfully exported features to: {output_path}")
        return os.path.abspath(output_path)

    except Exception as e:
        logging.error(f"Failed to export feature table to {output_path}: {e}")
        raise IOError(f"Parquet export failed: {e}") from e