import os
import pandas as pd
import pytest


DATA_PATH = "data/processed/transactions_articles_clean.csv"


@pytest.fixture(autouse=True)
def check_data_exists():
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Processed dataset '{DATA_PATH}' is not present. Run the ingestion pipeline first.")


def test_real_hm_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)

    assert len(df) == 180259
    assert len(df.columns) == 29

    missing_count = df.isna().sum().sum()

    assert missing_count == 0


def test_real_hm_dataset_is_not_empty():
    df = pd.read_csv(DATA_PATH)

    assert not df.empty


def test_real_hm_dataset_has_required_columns():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "customer_id",
        "article_id",
        "detail_desc",
    ]

    for column in required_columns:
        assert column in df.columns