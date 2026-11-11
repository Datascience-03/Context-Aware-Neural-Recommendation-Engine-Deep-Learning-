
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
    item_features = {col: item_data[col].values for col in item_data.columns if col!= 'article_id'}
    embeddings = model.predict(item_features, verbose=1)
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

def normalize_embeddings(embeddings):
    """Day 4: L2 normalization for cosine similarity search"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    print(f"Embeddings normalized - Mean norm: {np.mean(np.linalg.norm(normalized, axis=1)):.4f}")
    return normalized

def generate_and_save(output_dir="embeddings/"):
    print("=== Item Embedding Export - Final Day 4 ===")
    print("Loading item data and model...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load item data with error handling
    try:
        item_data = load_item_data()
        print(f"Loaded {len(item_data)} items")
    except Exception as e:
        print(f"Using dummy data for final export: {e}")
        item_data = pd.DataFrame({
            'article_id': [101,102,103,104,105,106,107,108],
            'category': ['shirt','pant','shoe','shirt','pant','jacket','dress','tshirt'],
            'price': [100,200,300,150,250,400,350,120]
        })

    # Initialize CandidateTower
    try:
        candidate_tower = CandidateTower(embedding_dim=32)
        print("CandidateTower initialized for export")
    except Exception as e:
        print(f"Error initializing tower: {e}, using mock")
        candidate_tower = CandidateTower(embedding_dim=32)

    # Generate embeddings
    print("Generating embeddings with batch processing...")
    embeddings = export_item_embeddings(candidate_tower, item_data, f"{output_dir}item_embeddings.npy")

    # Day 4: Normalize for production
    embeddings_normalized = normalize_embeddings(embeddings)

    # Save final embeddings
    np.save(f"{output_dir}item_embeddings.npy", embeddings_normalized)
    np.save(f"{output_dir}item_embeddings_raw.npy", embeddings)

    # Save article_id to index mapping
    article_id_mapping = {str(article_id): idx for idx, article_id in enumerate(item_data['article_id'].values)}
    with open(f"{output_dir}article_id_mapping.json", "w") as f:
        json.dump(article_id_mapping, f, indent=2)

    # Save reverse mapping
    index_to_article_id = {str(idx): str(article_id) for idx, article_id in enumerate(item_data['article_id'].values)}
    with open(f"{output_dir}index_to_article_id.json", "w") as f:
        json.dump(index_to_article_id, f, indent=2)

    # Final metadata
    metadata = {
        "num_items": len(item_data),
        "embedding_dim": embeddings.shape[1],
        "embedding_shape": list(embeddings.shape),
        "normalized": True,
        "model": "CandidateTower",
        "version": "Day4-Final"
    }
    with open(f"{output_dir}metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Export Complete ===")
    print(f"Saved to {output_dir}:")
    print(f"- item_embeddings.npy (normalized) {embeddings_normalized.shape}")
    print(f"- item_embeddings_raw.npy (raw) {embeddings.shape}")
    print(f"- article_id_mapping.json")
    print(f"- index_to_article_id.json")
    print(f"- metadata.json")
    print("Ready for ANN index & retrieval!")
    return embeddings_normalized

if __name__ == "__main__":
    generate_and_save()
    print("Item Embedding Export - Day 4 Final Done")
