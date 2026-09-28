
import numpy as np
import json
import tensorflow as tf
from pathlib import Path
import pandas as pd
from candidate_tower import CandidateTower
from item_embeddings import load_item_data

def export_item_embeddings(model, item_data, output_path="item_embeddings.npy"):
    """Generate and save pre-computed item embeddings"""
    print("Generating item embeddings...")
    # Day 2: Generate embeddings using CandidateTower
    item_features = {col: item_data[col].values for col in item_data.columns if col!= 'article_id'}
    embeddings = model.predict(item_features, verbose=1)
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

def generate_and_save(output_dir="embeddings/"):
    print("Loading item data and model...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load item data
    try:
        item_data = load_item_data()
        print(f"Loaded {len(item_data)} items")
    except Exception as e:
        print(f"Using dummy data: {e}")
        item_data = pd.DataFrame({
            'article_id': [101,102,103,104,105],
            'category': ['shirt','pant','shoe','shirt','pant'],
            'price': [100,200,300,150,250]
        })

    # Initialize CandidateTower
    candidate_tower = CandidateTower(embedding_dim=32)
    print("CandidateTower initialized for export")

    # Day 3: Batch processing + Save with mapping
    print("Day 3: Generating embeddings with batch processing...")
    embeddings = export_item_embeddings(candidate_tower, item_data, f"{output_dir}item_embeddings.npy")

    # Save embeddings
    np.save(f"{output_dir}item_embeddings.npy", embeddings)

    # Day 3: Save article_id to index mapping
    article_id_mapping = {str(article_id): idx for idx, article_id in enumerate(item_data['article_id'].values)}
    with open(f"{output_dir}article_id_mapping.json", "w") as f:
        json.dump(article_id_mapping, f, indent=2)

    # Day 3: Save metadata
    metadata = {
        "num_items": len(item_data),
        "embedding_dim": embeddings.shape[1],
        "embedding_shape": list(embeddings.shape)
    }
    with open(f"{output_dir}metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved to {output_dir} -> item_embeddings.npy, article_id_mapping.json, metadata.json")
    return candidate_tower

if __name__ == "__main__":
    generate_and_save()
    print("Item Embedding Export - Day 3 Done")