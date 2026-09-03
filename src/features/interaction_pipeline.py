import logging
from dataclasses import dataclass
from typing import List, Optional, Set
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaConfig:
    """
    Standard schema alignment for Two-Tower retrieval:
    - Query Tower (Member 1): User & Context features
    - Candidate Tower (Member 2): Item features
    """
    # Raw log source column names (Input)
    RAW_USER_ID: str = "visitor_id"
    RAW_ITEM_ID: str = "product_id"
    RAW_EVENT_TYPE: str = "action"
    RAW_TIMESTAMP: str = "event_time"
    RAW_USER_DEVICE: str = "device_type"

    # Aligned schema names (Output shared with Towers)
    QUERY_USER_ID: str = "user_id"          # Required by Member 1
    QUERY_USER_DEVICE: str = "user_device"  # Context feature for Member 1
    CANDIDATE_ITEM_ID: str = "item_id"      # Required by Member 2
    TIMESTAMP: str = "timestamp"


class InteractionLoader:
    """Loads, validates, and extracts positive interaction pairs for Two-Tower models."""

    def __init__(
        self,
        config: SchemaConfig = SchemaConfig(),
        positive_event_types: Optional[Set[str]] = None
    ):
        self.config = config
        # Define what constitutes a "positive" implicit feedback interaction
        self.positive_event_types = positive_event_types or {
            "click",
            "add_to_cart",
            "purchase",
            "like"
        }

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Loads interaction logs from CSV or Parquet."""
        logger.info(f"Loading raw logs from: {file_path}")
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)

    def filter_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out invalid records and keeps only valid positive samples."""
        initial_count = len(df)
        logger.info(f"Starting data validation. Initial rows: {initial_count}")

        # 1. Drop records with missing critical identifiers
        critical_cols = [self.config.RAW_USER_ID, self.config.RAW_ITEM_ID, self.config.RAW_EVENT_TYPE]
        df = df.dropna(subset=critical_cols).copy()

        # 2. Filter for designated positive interactions
        df[self.config.RAW_EVENT_TYPE] = df[self.config.RAW_EVENT_TYPE].astype(str).str.lower().str.strip()
        df = df[df[self.config.RAW_EVENT_TYPE].isin(self.positive_event_types)]

        # 3. Clean IDs (ensure strings, trim whitespace)
        df[self.config.RAW_USER_ID] = df[self.config.RAW_USER_ID].astype(str).str.strip()
        df[self.config.RAW_ITEM_ID] = df[self.config.RAW_ITEM_ID].astype(str).str.strip()

        # 4. Remove empty strings
        df = df[(df[self.config.RAW_USER_ID] != "") & (df[self.config.RAW_ITEM_ID] != "")]

        # 5. Deduplicate identical user-item interactions occurring on the same timestamp
        dedup_cols = [self.config.RAW_USER_ID, self.config.RAW_ITEM_ID]
        if self.config.RAW_TIMESTAMP in df.columns:
            dedup_cols.append(self.config.RAW_TIMESTAMP)
        df = df.drop_duplicates(subset=dedup_cols)

        logger.info(
            f"Filtered positive interactions: {len(df)} rows retained "
            f"({initial_count - len(df)} rows dropped)."
        )
        return df

    def align_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames and structures columns to align with:
        - Member 1 (Query Tower): user_id, user_device
        - Member 2 (Candidate Tower): item_id
        """
        logger.info("Aligning schema with Query & Candidate Tower standards...")

        rename_map = {
            self.config.RAW_USER_ID: self.config.QUERY_USER_ID,
            self.config.RAW_ITEM_ID: self.config.CANDIDATE_ITEM_ID,
            self.config.RAW_TIMESTAMP: self.config.TIMESTAMP,
            self.config.RAW_USER_DEVICE: self.config.QUERY_USER_DEVICE,
        }

        # Keep only mapped columns that exist in the dataframe
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        aligned_df = df.rename(columns=rename_map)

        # Standard column order
        priority_cols = [
            self.config.QUERY_USER_ID,
            self.config.CANDIDATE_ITEM_ID,
            self.config.TIMESTAMP,
            self.config.QUERY_USER_DEVICE,
        ]
        final_cols = [c for c in priority_cols if c in aligned_df.columns]
        
        # Add any remaining feature columns
        remaining_cols = [c for c in aligned_df.columns if c not in final_cols and c != self.config.RAW_EVENT_TYPE]
        
        return aligned_df[final_cols + remaining_cols]

    def process(self, input_path: Optional[str] = None, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """End-to-end pipeline run."""
        if raw_df is None and input_path is not None:
            raw_df = self.load_raw_data(input_path)
        elif raw_df is None:
            raise ValueError("Must provide either `input_path` or `raw_df`.")

        cleaned_df = self.filter_and_clean(raw_df)
        aligned_df = self.align_schema(cleaned_df)
        return aligned_df


# =====================================================================
# Test / Demo Verification
# =====================================================================
if __name__ == "__main__":
    # 1. Create dummy raw interaction logs with dirty edge cases
    mock_logs = pd.DataFrame({
        "visitor_id": ["u101", "u102", "u103", None,   "u101", "u104", "  "],
        "product_id": ["p501", "p502", "p501", "p503", "p501", "p504", "p505"],
        "action":     ["click", "impression", "purchase", "click", "click", "like", "click"],
        "event_time": [
            "2023-10-01 10:00:00",
            "2023-10-01 10:01:00",
            "2023-10-01 10:02:00",
            "2023-10-01 10:03:00",
            "2023-10-01 10:00:00",  # Duplicate row
            "2023-10-01 10:05:00",
            "2023-10-01 10:06:00"
        ],
        "device_type": ["mobile", "desktop", "mobile", "ios", "mobile", "android", "web"]
    })

    print("\n--- RAW LOGS SAMPLE ---")
    print(mock_logs)

    # 2. Run Pipeline
    pipeline = InteractionLoader()
    positive_pairs = pipeline.process(raw_df=mock_logs)

    print("\n--- EXTRACTED POSITIVE (USER, ITEM) PAIRS ---")
    print(positive_pairs)
    print("\nSchema Datatypes:")
    print(positive_pairs.dtypes)