# Day 1 - Setup item embedding export skeleton
# Day 2 - Implement embedding generation using CandidateTower
import numpy as np
import json
import tensorflow as tf
from candidate_tower import CandidateTower
from item_embeddings import load_item_data

def export_item_embeddings(model, item_data, output_path="item_embeddings.npy"):
    """Generate and save pre-computed item embeddings"""
    print("Generating item embeddings...")
    # Day 2: Generate embeddings using CandidateTower
    item_features = {col: item_data[col].values for col in item_data.columns if col!= 'article_id'}
    embeddings = model.predict(item_features)
    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

def generate_and_save():
    print("Loading item data and model...")
    candidate_tower = CandidateTower(embedding_dim=32)
    print("CandidateTower initialized for export")
    return candidate_tower

if __name__ == "__main__":
    generate_and_save()
    print("Item Embedding Export - Day 2 Done")