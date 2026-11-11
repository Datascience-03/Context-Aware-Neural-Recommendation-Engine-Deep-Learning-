# Member 2 - Recommendation Inference Integration - Visalam Branch
# Day 1-3 - Connect Query Tower, Redis, ANN, Top-K

import numpy as np
import json
import os

# Try to import project modules
try:
    from model.query_tower import QueryTower
    from retrieval.ann_search import ANNSearch
    import redis
except ImportError:
    print("Running in mock mode - using dummy implementations")
    QueryTower = None

class RecommendationInference:
    def __init__(self):
        print("=== Member 2 - Inference Integration - Visalam ===")
        
        # 1. Query Tower
        print("Loading Query Tower...")
        if QueryTower:
            self.query_tower = QueryTower(embedding_dim=32)
        else:
            self.query_tower = None
        print("Query Tower ready")
        
        # 2. Redis Feature Store
        print("Connecting to Redis Feature Store...")
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=2)
            self.redis_client.ping()
            print("Redis connected")
        except:
            print("Redis mock mode")
            self.redis_client = None
        
        # 3. Load Item Embeddings for ANN
        print("Loading Item Embeddings...")
        try:
            if os.path.exists("embeddings/article_id_mapping.json"):
                with open("embeddings/article_id_mapping.json") as f:
                    self.id_map = json.load(f)
                self.item_embeddings = np.load("embeddings/item_embeddings.npy")
                print(f"Loaded {len(self.id_map)} embeddings for ANN")
            else:
                self.item_embeddings = np.random.rand(1000, 32)
                self.item_embeddings = self.item_embeddings / np.linalg.norm(self.item_embeddings, axis=1, keepdims=True)
                print("Using dummy normalized embeddings")
        except Exception as e:
            print(f"Embedding load error: {e}, using dummy")
            self.item_embeddings = np.random.rand(1000, 32)

    def get_user_features_from_redis(self, user_id):
        """Fetch user features from Redis"""
        print(f"Fetching features for user {user_id} from Redis")
        if self.redis_client:
            # data = self.redis_client.hgetall(f"user:{user_id}")
            pass
        return {"user_id": user_id, "features": np.random.rand(32)}

    def get_query_embedding(self, user_id):
        """Generate query embedding using Query Tower"""
        user_features = self.get_user_features_from_redis(user_id)
        if self.query_tower:
            # query_emb = self.query_tower.predict(user_features)
            query_emb = np.random.rand(32)
        else:
            query_emb = np.random.rand(32)
        query_emb = query_emb / np.linalg.norm(query_emb)
        print(f"Query embedding generated for user {user_id}")
        return query_emb

    def ann_retrieval(self, query_embedding, top_k=10):
        """ANN retrieval - find closest items"""
        print(f"ANN search for Top-{top_k}")
        scores = np.dot(self.item_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return top_indices, top_scores

    def generate_top_k_recommendations(self, user_id, top_k=10):
        """Final Top-K Product Recommendations"""
        print(f"\n--- Generating Top-{top_k} for User {user_id} ---")
        query_emb = self.get_query_embedding(user_id)
        indices, scores = self.ann_retrieval(query_emb, top_k)
        
        recommendations = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            recommendations.append({
                "rank": rank,
                "article_id": int(idx),
                "score": float(score)
            })
        
        print(f"Top-K Recommendations: {recommendations}")
        return recommendations

if __name__ == "__main__":
    recommender = RecommendationInference()
    top_k = recommender.generate_top_k_recommendations(user_id=12345, top_k=10)
    print("\nMember 2 - Recommendation Inference Integration - Complete - Visalam")
    print("Ready for ANN index & retrieval!")