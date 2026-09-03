"""
Day 1: Interaction Data Loading & Positive Sample Extraction
Aligned with Member 1 (Query Tower) and Member 2 (Candidate Tower) schemas.
Supports both standard e-commerce interaction logs and H&M transactions format.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaConfig:
    # Target column names aligned with Tower specifications
    QUERY_USER_ID: str = "user_id"        # Member 1 (Query Tower)
    CANDIDATE_ITEM_ID: str = "item_id"    # Member 2 (Candidate Tower)
    TIMESTAMP: str = "timestamp"          # Shared / Temporal Split

    # H&M transaction schema mapping
    HM_USER_COL: str = "customer_id"
    HM_ITEM_COL: str = "article_id"
    HM_DATE_COL: str = "t_dat"

    # Generic interaction log column mapping
    GENERIC_USER_COL: str = "visitor_id"
    GENERIC_ITEM_COL: str = "product_id"
    GENERIC_ACTION_COL: str = "action"
    GENERIC_DATE_COL: str = "event_time"


class InteractionLoader:
    def __init__(
        self,
        config: SchemaConfig = SchemaConfig(),
        positive_actions: Optional[Set[str]] = None
    ):
        self.config = config
        self.positive_actions = positive_actions or {"purchase", "click", "add_to_cart", "like"}

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading raw interactions from: {file_path}")
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def extract_positive_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans, filters positive interactions, and aligns schema."""
        initial_len = len(df)
        logger.info(f"Input records count: {initial_len}")

        cols = set(df.columns)

        # Detect schema style (H&M transaction format vs. standard event logs)
        if self.config.HM_USER_COL in cols and self.config.HM_ITEM_COL in cols:
            # All transactions in H&M are purchases -> positive implicit samples
            user_col = self.config.HM_USER_COL
            item_col = self.config.HM_ITEM_COL
            time_col = self.config.HM_DATE_COL if self.config.HM_DATE_COL in cols else None
        elif self.config.GENERIC_USER_COL in cols and self.config.GENERIC_ITEM_COL in cols:
            user_col = self.config.GENERIC_USER_COL
            item_col = self.config.GENERIC_ITEM_COL
            time_col = self.config.GENERIC_DATE_COL if self.config.GENERIC_DATE_COL in cols else None

            # Filter for positive events
            if self.config.GENERIC_ACTION_COL in cols:
                df[self.config.GENERIC_ACTION_COL] = (
                    df[self.config.GENERIC_ACTION_COL].astype(str).str.lower().str.strip()
                )
                df = df[df[self.config.GENERIC_ACTION_COL].isin(self.positive_actions)]
        else:
            # Fallback to direct names if already partially aligned
            user_col = "user_id" if "user_id" in cols else df.columns[0]
            item_col = "item_id" if "item_id" in cols else df.columns[1]
            time_col = "timestamp" if "timestamp" in cols else None

        # 1. Drop missing user/item IDs
        df = df.dropna(subset=[user_col, item_col]).copy()

        # 2. String conversion & strip whitespace
        df[user_col] = df[user_col].astype(str).str.strip()
        df[item_col] = df[item_col].astype(str).str.strip()
        df = df[(df[user_col] != "") & (df[item_col] != "")]

        # 3. Rename columns to align with Query and Candidate Towers
        rename_dict = {
            user_col: self.config.QUERY_USER_ID,
            item_col: self.config.CANDIDATE_ITEM_ID,
        }
        if time_col and time_col in df.columns:
            rename_dict[time_col] = self.config.TIMESTAMP

        df = df.rename(columns=rename_dict)

        # 4. Remove duplicate interactions on the same timestamp
        dedup_subset = [self.config.QUERY_USER_ID, self.config.CANDIDATE_ITEM_ID]
        if self.config.TIMESTAMP in df.columns:
            dedup_subset.append(self.config.TIMESTAMP)
        df = df.drop_duplicates(subset=dedup_subset)

        # 5. Output ordered columns
        keep_cols = [self.config.QUERY_USER_ID, self.config.CANDIDATE_ITEM_ID]
        if self.config.TIMESTAMP in df.columns:
            keep_cols.append(self.config.TIMESTAMP)

        result_df = df[keep_cols].reset_index(drop=True)
        logger.info(f"Extracted {len(result_df)} positive pairs ({initial_len - len(result_df)} dropped).")
        return result_df


if __name__ == "__main__":
    # Quick sanity check with sample data
    test_data = pd.DataFrame({
        "customer_id": ["u1", "u2", "u1", None, "u3"],
        "article_id": ["0858883001", "0858883002", "0858883001", "0858883003", "0858883004"],
        "t_dat": ["2020-09-01", "2020-09-01", "2020-09-01", "2020-09-02", "2020-09-02"]
    })

    loader = InteractionLoader()
    positive_pairs = loader.extract_positive_pairs(test_data)
    print("\n--- Aligned Positive Samples Output ---")
    print(positive_pairs.head())