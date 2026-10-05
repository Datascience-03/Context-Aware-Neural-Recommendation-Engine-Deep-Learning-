import numpy as np

from src.feature_store.redis_item_vector_store import RedisItemVectorStore


TEST_PREFIX = "test_item_vector:"


def create_store():
    store = RedisItemVectorStore(
        redis_url="redis://localhost:6379/0",
        key_prefix=TEST_PREFIX,
    )
    store.clear_all_item_vectors()
    return store


def test_redis_connection():
    store = create_store()

    try:
        assert store.ping() is True
    finally:
        store.clear_all_item_vectors()
        store.close()


def test_store_and_retrieve_item_vector():
    store = create_store()

    try:
        item_id = "test_1001"
        vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        store.store_item_vector(item_id, vector)

        retrieved = store.get_item_vector(item_id)

        assert retrieved is not None
        assert retrieved.dtype == np.float32
        assert retrieved.shape == (4,)
        assert np.allclose(retrieved, vector)
    finally:
        store.clear_all_item_vectors()
        store.close()


def test_item_exists():
    store = create_store()

    try:
        item_id = "test_1002"
        vector = np.ones(64, dtype=np.float32)

        assert store.item_exists(item_id) is False

        store.store_item_vector(item_id, vector)

        assert store.item_exists(item_id) is True
    finally:
        store.clear_all_item_vectors()
        store.close()


def test_store_multiple_embeddings():
    store = create_store()

    try:
        item_ids = np.array(
            ["test_2001", "test_2002", "test_2003"]
        )

        embeddings = np.array(
            [
                np.ones(64, dtype=np.float32),
                np.full(64, 2.0, dtype=np.float32),
                np.full(64, 3.0, dtype=np.float32),
            ],
            dtype=np.float32,
        )

        stored = store.store_embeddings(
            item_ids,
            embeddings,
            batch_size=2,
        )

        assert stored == 3
        assert store.count_item_vectors() == 3

        for item_id, expected in zip(item_ids, embeddings):
            retrieved = store.get_item_vector(item_id)

            assert retrieved is not None
            assert retrieved.shape == (64,)
            assert np.allclose(retrieved, expected)
    finally:
        store.clear_all_item_vectors()
        store.close()


def test_mismatched_ids_and_embeddings():
    store = create_store()

    try:
        item_ids = np.array(["test_3001", "test_3002"])

        embeddings = np.ones(
            (3, 64),
            dtype=np.float32,
        )

        try:
            store.store_embeddings(item_ids, embeddings)
            assert False, "Expected ValueError"
        except ValueError:
            pass
    finally:
        store.clear_all_item_vectors()
        store.close()


def test_delete_item_vector():
    store = create_store()

    try:
        item_id = "test_4001"
        vector = np.ones(64, dtype=np.float32)

        store.store_item_vector(item_id, vector)

        assert store.item_exists(item_id) is True

        deleted = store.delete_item_vector(item_id)

        assert deleted is True
        assert store.item_exists(item_id) is False
        assert store.get_item_vector(item_id) is None
    finally:
        store.clear_all_item_vectors()
        store.close()


def test_missing_item_returns_none():
    store = create_store()

    try:
        retrieved = store.get_item_vector("does_not_exist")

        assert retrieved is None
    finally:
        store.clear_all_item_vectors()
        store.close()