"""
Module for contextual feature engineering:
- Time of purchase features
- User recency features
- Product popularity over time
"""

import pandas as pd
import numpy as np


class ContextualFeatureExtractor:
    def __init__(self, date_col: str = "t_dat"):
        self.date_col = date_col

    def inspect_date_range(self, df: pd.DataFrame) -> dict:
        """Inspect start, end date, and total days in transactions."""
        min_date = pd.to_datetime(df[self.date_col]).min()
        max_date = pd.to_datetime(df[self.date_col]).max()
        return {
            "start_date": min_date.strftime("%Y-%m-%d"),
            "end_date": max_date.strftime("%Y-%m-%d"),
            "total_days": (max_date - min_date).days + 1
        }