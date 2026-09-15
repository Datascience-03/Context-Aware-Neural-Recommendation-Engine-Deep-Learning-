
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.contextual_features import (
    extract_calendar_features,
    add_cyclical_encoding,
    generate_recency_frequency_features,
    compute_product_popularity_over_time,
)


def test_calendar_features():
    df = pd.DataFrame({
        "t_dat": ["2024-01-15", "2024-07-20"]
    })

    result = extract_calendar_features(df)

    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns
    assert "month" in result.columns
    assert "quarter" in result.columns
    assert "season" in result.columns

    assert result.loc[0, "month"] == 1
    assert result.loc[0, "quarter"] == 1
    assert result.loc[0, "season"] == "Winter"

    assert result.loc[1, "month"] == 7
    assert result.loc[1, "quarter"] == 3
    assert result.loc[1, "season"] == "Summer"


def test_cyclical_features():
    df = pd.DataFrame({
        "day_of_week": [1, 4, 7],
        "month": [1, 6, 12]
    })

    result = add_cyclical_encoding(df)

    expected = [
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
    ]

    for column in expected:
        assert column in result.columns
        assert result[column].notna().all()

    numeric_values = result[expected].to_numpy()
    assert np.isfinite(numeric_values).all()


def test_recency_frequency_features():
    df = pd.DataFrame({
        "customer_id": ["U1", "U1", "U1", "U2"],
        "t_dat": [
            "2024-01-01",
            "2024-01-05",
            "2024-01-10",
            "2024-02-01",
        ],
    })

    result = generate_recency_frequency_features(df)

    assert "purchase_sequence" in result.columns
    assert "days_since_last_purchase" in result.columns
    assert "days_since_first_purchase" in result.columns
    assert "avg_inter_purchase_days" in result.columns
    assert "is_first_purchase" in result.columns

    u1 = result[result["customer_id"] == "U1"].sort_values("t_dat")

    assert u1["purchase_sequence"].tolist() == [1, 2, 3]
    assert u1["days_since_last_purchase"].tolist() == [-1.0, 4.0, 5.0]
    assert u1["is_first_purchase"].tolist() == [1, 0, 0]

    assert result.isna().sum().sum() == 0


def test_popularity_feature_exists_and_has_no_missing_values():
    df = pd.DataFrame({
        "article_id": [1001, 1001, 1002],
        "t_dat": [
            "2024-01-01",
            "2024-01-10",
            "2024-01-10",
        ],
    })

    result = compute_product_popularity_over_time(df)

    assert "popularity_over_time" in result.columns
    assert result["popularity_over_time"].notna().all()
    assert np.isfinite(result["popularity_over_time"]).all()


def test_real_hm_contextual_features():
    path = "data/processed/contextual_features.csv"
    df = pd.read_csv(path)

    expected_columns = [
        "day_of_week",
        "is_weekend",
        "month",
        "quarter",
        "season",
        "day_of_week_sin",
        "day_of_week_cos",
        "month_sin",
        "month_cos",
        "days_since_last_purchase",
        "purchase_sequence",
        "days_since_first_purchase",
        "avg_inter_purchase_days",
        "is_first_purchase",
        "popularity_over_time",
    ]

    for column in expected_columns:
        assert column in df.columns, f"Missing contextual feature: {column}"

    assert len(df) == 200000
    assert df[expected_columns].isna().sum().sum() == 0

    numeric_columns = [
        column
        for column in expected_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    assert np.isfinite(df[numeric_columns].to_numpy()).all()
