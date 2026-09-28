import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.missing_values import handle_missing_values
from src.data_processing.cold_start import (
    get_popular_items,
    get_demographic_popular_items,
    handle_user_cold_start,
    handle_item_cold_start,
)


def create_test_data():
    transactions = pd.DataFrame({
        "customer_id": ["U1", "U1", "U2", "U3", "U3", "U3"],
        "article_id": [1001, 1002, 1001, 1003, 1003, 1001],
    })

    customers = pd.DataFrame({
        "customer_id": ["U1", "U2", "U3", "U_NEW"],
        "age": [25, 30, 26, 45],
    })

    articles = pd.DataFrame({
        "article_id": [1001, 1002, 1003, 1004],
        "product_group_name": [
            "Upper wear",
            "Upper wear",
            "Lower wear",
            "Lower wear",
        ],
    })

    return transactions, customers, articles


def test_missing_description_is_filled():
    df = pd.DataFrame({
        "article_id": [1, 2, 3],
        "detail_desc": ["Shirt", None, "Trousers"],
    })

    result = handle_missing_values(df)

    assert result["detail_desc"].isna().sum() == 0
    assert result.loc[1, "detail_desc"] == "No description available"


def test_no_missing_values_remain():
    df = pd.DataFrame({
        "article_id": [1, 2],
        "detail_desc": ["Shirt", "Trousers"],
    })

    result = handle_missing_values(df)

    assert result.isna().sum().sum() == 0


def test_get_popular_items():
    transactions, _, _ = create_test_data()

    result = get_popular_items(transactions, limit=2)

    assert result == [1001, 1003]


def test_demographic_popular_items():
    transactions, customers, _ = create_test_data()

    result = get_demographic_popular_items(
        transactions,
        customers,
        age=27,
        limit=2,
    )

    assert result == [1001, 1003]


def test_new_user_cold_start():
    transactions, customers, _ = create_test_data()

    result = handle_user_cold_start(
        "U_NEW",
        transactions,
        customers,
        limit=2,
    )

    assert result == [1001, 1003]


def test_existing_user_is_not_cold_start():
    transactions, customers, _ = create_test_data()

    result = handle_user_cold_start(
        "U1",
        transactions,
        customers,
        limit=2,
    )

    assert result is None


def test_new_item_cold_start():
    transactions, _, articles = create_test_data()

    result = handle_item_cold_start(
        1004,
        articles,
        transactions,
        limit=2,
    )

    assert result == [1003]


def test_existing_item_is_not_cold_start():
    transactions, _, articles = create_test_data()

    result = handle_item_cold_start(
        1001,
        articles,
        transactions,
        limit=2,
    )

    assert result is None


def test_empty_transactions():
    _, customers, articles = create_test_data()

    empty_transactions = pd.DataFrame(
        columns=["customer_id", "article_id"]
    )

    user_result = handle_user_cold_start(
        "U_NEW",
        empty_transactions,
        customers,
        limit=2,
    )

    item_result = handle_item_cold_start(
        1004,
        articles,
        empty_transactions,
        limit=2,
    )

    assert user_result == []
    assert item_result == []


def test_unknown_customer_falls_back_to_popularity():
    transactions, customers, _ = create_test_data()

    result = handle_user_cold_start(
        "UNKNOWN_USER",
        transactions,
        customers,
        limit=2,
    )

    assert result == [1001, 1003]