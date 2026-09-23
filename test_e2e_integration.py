import numpy as np
from vector_store import ItemVectorStore


def run_e2e_test():
    print("=== Starting Day 7: E2E Integration Test ===")

    # 1. Initialize Store
    store = ItemVectorStore()
    assert store.ping() is True, "Failed to connect to vector store."
    print("✔ Storage engine initialized and responding.")

    # 2. Member 2 Simulation: Generate mock item embeddings
    embedding_dim = 64
    mock_catalog = {
        "item_101": np.random.randn(embedding_dim).astype(np.float32),
        "item_102": np.random.randn(embedding_dim).astype(np.float32),
        "item_103": np.random.randn(embedding_dim).astype(np.float32),
    }
    print(f"✔ Member 2: Generated embeddings for {len(mock_catalog)} items.")

    # 3. Member 4 Simulation: Store single & batch vectors
    store.set_vector("item_101", mock_catalog["item_101"])
    store.set_vectors_batch({
        "item_102": mock_catalog["item_102"],
        "item_103": mock_catalog["item_103"]
    })
    print("✔ Member 4: Stored vectors successfully.")

    # 4. Member 5 / Downstream Model: Query single vector
    vec_101 = store.get_vector("item_101")
    assert vec_101 is not None, "Error: item_101 not found"
    assert np.allclose(vec_101, mock_catalog["item_101"]), "Error: Data mismatch on item_101"
    print("✔ Member 5: Single vector retrieved with matching precision.")

    # 5. Member 5 / Downstream Model: Query batch with missing key
    batch_result = store.get_vectors_batch(["item_101", "item_102", "item_nonexistent"])
    assert len(batch_result) == 2, f"Expected 2 results, got {len(batch_result)}"
    assert "item_nonexistent" not in batch_result, "Missing ID should not be returned"
    assert np.allclose(batch_result["item_102"], mock_catalog["item_102"]), "Error: Data mismatch on item_102"
    print("✔ Member 5: Batch vector retrieval verified.")

    # 6. Cleanup
    for item_id in mock_catalog:
        store.delete_vector(item_id)
    print("✔ Cleanup finished.")

    print("\n🎉 ALL DAY 7 INTEGRATION TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_e2e_test()