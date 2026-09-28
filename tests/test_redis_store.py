import os
import pytest
from src.feature_store.redis_store import RedisUserProfileStore

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture
def store():
    feature_store = RedisUserProfileStore(
        redis_url=REDIS_URL,
        key_prefix="test:user:"
    )

    try:
        feature_store.ping()
    except Exception:
        pytest.skip("Redis server is not available.")

    yield feature_store
    feature_store.close()


@pytest.fixture
def user_profile():
    return {
        "customer_id_idx": 10,
        "month_sin": 0.5,
        "month_cos": 0.866,
        "day_of_week_sin": 0.0,
        "day_of_week_cos": 1.0,
        "is_weekend": 1.0,
        "days_since_last_purchase": 5.0,
        "purchase_sequence": 12.0,
    }


def test_redis_connection(store):
    assert store.ping() is True


def test_store_and_get_user_profile(store, user_profile):
    customer_id = "test_001"

    store.delete_user_profile(customer_id)
    assert store.store_user_profile(customer_id, user_profile)

    result = store.get_user_profile(customer_id)

    assert result is not None
    assert result["customer_id_idx"] == 10
    assert result["month_sin"] == pytest.approx(0.5)
    assert result["purchase_sequence"] == pytest.approx(12.0)

    store.delete_user_profile(customer_id)


def test_user_exists(store, user_profile):
    customer_id = "test_002"

    store.delete_user_profile(customer_id)

    assert store.user_exists(customer_id) is False

    store.store_user_profile(customer_id, user_profile)

    assert store.user_exists(customer_id) is True

    store.delete_user_profile(customer_id)


def test_update_user_profile(store, user_profile):
    customer_id = "test_003"

    store.delete_user_profile(customer_id)
    store.store_user_profile(customer_id, user_profile)

    store.update_user_profile(
        customer_id,
        {
            "days_since_last_purchase": 2.0,
            "purchase_sequence": 20.0,
        }
    )

    result = store.get_user_profile(customer_id)

    assert result["days_since_last_purchase"] == pytest.approx(2.0)
    assert result["purchase_sequence"] == pytest.approx(20.0)

    store.delete_user_profile(customer_id)


def test_get_missing_user_returns_none(store):
    customer_id = "user_that_does_not_exist"

    store.delete_user_profile(customer_id)

    result = store.get_user_profile(customer_id)

    assert result is None


def test_delete_user_profile(store, user_profile):
    customer_id = "test_004"

    store.delete_user_profile(customer_id)
    store.store_user_profile(customer_id, user_profile)

    assert store.user_exists(customer_id) is True

    deleted = store.delete_user_profile(customer_id)

    assert deleted is True
    assert store.user_exists(customer_id) is False


def test_get_query_features(store, user_profile):
    customer_id = "test_005"

    store.delete_user_profile(customer_id)
    store.store_user_profile(customer_id, user_profile)

    query_features = store.get_query_features(customer_id)

    expected_features = {
        "customer_id_idx",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "days_since_last_purchase",
        "purchase_sequence",
    }

    assert set(query_features.keys()) == expected_features

    store.delete_user_profile(customer_id)