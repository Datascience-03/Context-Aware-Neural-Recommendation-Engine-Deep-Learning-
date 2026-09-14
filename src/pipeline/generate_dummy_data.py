# generate_dummy_data.py
import numpy as np
import pandas as pd

def create_sample_dataset(filepath="items_vectors.parquet", num_items=1000, dim=128):
    item_ids = [f"item_{i}" for i in range(num_items)]
    # Generate random 128-dimensional float32 vectors
    vectors = [np.random.rand(dim).astype(np.float32).tolist() for _ in range(num_items)]

    df = pd.DataFrame({
        "item_id": item_ids,
        "vector": vectors
    })
    df.to_parquet(filepath, index=False)
    print(f"Generated {num_items} items with {dim}-d vectors -> {filepath}")

if __name__ == "__main__":
    create_sample_dataset()