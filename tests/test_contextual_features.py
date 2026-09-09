import pytest
import numpy as np
import pandas as pd
from src.features.trend_features import (
    compute_popularity_trend_ratio,
    compute_exponential_decay_score,
)
from src.features.memory_optimizer import reduce_memory_usage


@pytest.fixture
def sample_transactions():
    ref_date = pd.Timestamp("2024-03-31")
    data = [
        # Item 1: High recent surge (trending) -> 3 sales in 7d, 4 total in 30d
        ("item_1", ref_date - pd.Timedelta(days=1)),
        ("item_1", ref_date - pd.Timedelta(days=2)),
        ("item_1", ref_date - pd.Timedelta(days=5)),
        ("item_1", ref_date - pd.Timedelta(days=20)),
        # Item 2: Old popularity (decaying) -> 0 sales in 7d, 3 sales in 30d
        ("item_2", ref_date - pd.Timedelta(days=15)),
        ("item_2", ref_date - pd.Timedelta(days=20)),
        ("item_2", ref_date - pd.Timedelta(days=25)),
        # Item 3: Brand new item -> 1 sale today
        ("item_3", ref_date),
    ]
    return pd.DataFrame(data, columns=["item_id", "transaction_date"]), ref_date


def test_trend_ratio_calculation(sample_transactions):
    df, ref_date = sample_transactions
    res = compute_popularity_trend_ratio(df, ref_date)

    # Item 1: 3 / ((4 / 4) + 1e-5) ~= 3.0
    item1_trend = res.loc[res["item_id"] == "item_1", "trend_ratio"].values[0]
    assert np.isclose(item1_trend, 3.0, atol=1e-3)

    # Item 2: 0 / ((3 / 4) + 1e-5) == 0.0
    item2_trend = res.loc[res["item_id"] == "item_2", "trend_ratio"].values[0]
    assert np.isclose(item2_trend, 0.0, atol=1e-3)


def test_exponential_decay_score(sample_transactions):
    df, ref_date = sample_transactions
    res = compute_exponential_decay_score(df, ref_date, half_life_days=7.0)

    # Item 3 bought on reference date (delta_t = 0) -> score must be exp(0) = 1.0
    item3_score = res.loc[res["item_id"] == "item_3", "decay_score"].values[0]
    assert np.isclose(item3_score, 1.0, atol=1e-5)

    # Item 1 (recent) should have higher decay score than Item 2 (older)
    item1_score = res.loc[res["item_id"] == "item_1", "decay_score"].values[0]
    item2_score = res.loc[res["item_id"] == "item_2", "decay_score"].values[0]
    assert item1_score > item2_score


def test_memory_optimization():
    df = pd.DataFrame({
        "int_small": [1, 2, 3],
        "int_medium": [1000, 2000, 3000],
        "float_val": [1.123456789, 2.987654321, 3.5],
    })

    optimized = reduce_memory_usage(df)

    assert optimized["int_small"].dtype == np.int8
    assert optimized["int_medium"].dtype == np.int16
    assert optimized["float_val"].dtype == np.float32