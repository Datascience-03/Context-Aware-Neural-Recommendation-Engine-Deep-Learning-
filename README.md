# Context-Aware Neural Recommendation Engine (Deep Learning)

An end-to-end, production-grade deep learning recommendation system based on a **Two-Tower (Dual-Encoder)** neural architecture. The engine incorporates real-time temporal and user context, Approximate Nearest Neighbor (ANN) vector indexing, Redis-backed feature storage, and a low-latency FastAPI inference service.

---

## 🏗️ System Architecture

```
                      ┌──────────────────────────────────────┐
                      │            User & Context            │
                      │  (User ID, Recency, Cyclical Time)   │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                               ┌───────────────────┐
                               │    Query Tower    │
                               │   (Dense + L2)    │
                               └─────────┬─────────┘
                                         │  (64-dim query vector)
                                         ▼
                      ┌──────────────────────────────────────┐
                      │    Fast Vector Retrieval (ANN)       │
                      │   Exact / IVF / HNSW / ScaNN Index   │
                      └──────────────────▲───────────────────┘
                                         │  (Item vectors)
                               ┌─────────┴─────────┐
                               │  Candidate Tower  │
                               │   (Dense + L2)    │
                               └─────────▲─────────┘
                                         │
                      ┌──────────────────┴───────────────────┐
                      │            Item Metadata             │
                      │ (Article ID, Product/Colour Groups)  │
                      └──────────────────────────────────────┘
```

1. **Query Tower**: Encodes the customer identifier and real-time context (cyclical month, cyclical day of week, weekend indicator, days since last purchase, purchase frequency sequence) into a shared $D$-dimensional normalized embedding space ($D=64$).
2. **Candidate Tower**: Encodes item attributes (article ID, product group, colour group, popularity over time) into the identical $D$-dimensional normalized embedding space.
3. **Training Objective**: In-batch sampled softmax / InfoNCE loss with temperature scaling ($\tau = 0.05$) to optimize cross-entropy over similarity matrices ($S = Q \cdot C^T$).
4. **Vector Retrieval (ANN)**: Fast nearest-neighbor index lookup using ExactSearch, Inverted File (IVF) with Voronoi partitioning, and HNSW proximity graphs.
5. **Feature Store**: Redis-backed cache for user context and item embeddings with automatic fallback to in-memory `fakeredis`.
6. **Serving Layer**: High-performance asynchronous FastAPI REST endpoints for real-time recommendation inference.

---

## 📁 Repository Structure

```text
├── configs/
│   └── config.yaml                     # Pipeline and model hyperparameters
├── data/
│   ├── processed/
│   │   └── model_export/               # Serialized production model weights & JSON architectures
│   └── reduced/                        # Sample offline datasets for testing and verification
├── docs/                               # Export documentation and component briefs
├── examples/
│   ├── day1_demo.py                    # Exact search baseline demonstration
│   ├── day2_demo.py                    # IVF index Voronoi partitioning demonstration
│   └── day3_demo.py                    # HNSW graph indexing & persistence demonstration
├── src/
│   ├── api/
│   │   └── main.py                     # FastAPI serving application (/health, /recommend, /metrics)
│   ├── features/
│   │   └── contextual_features.py      # Feature engineering: calendar, cyclical, recency, popularity
│   ├── feature_store/
│   │   ├── redis_store.py              # Redis user profile store
│   │   └── redis_item_vector_store.py  # Redis item vector store
│   ├── model/
│   │   ├── query_tower.py              # User & context Query Tower
│   │   ├── candidate_tower.py          # Item metadata Candidate Tower
│   │   ├── two_tower_model.py          # Unified Two-Tower model with in-batch loss
│   │   ├── train.py                    # Model training loop & embedding export
│   │   └── evaluate.py                 # Evaluation pipeline (Recall@K, NDCG@K, MRR)
│   ├── pipeline/
│   │   ├── airflow_dag.py              # Airflow DAG for scheduled batch processing
│   │   └── pyspark_pipeline.py         # PySpark distributed data transformation
│   └── retrieval/
│       └── ann_search.py               # Exact, IVF, and HNSW vector search indices
├── tests/                              # Comprehensive unit and integration test suite (181 tests)
├── benchmark.py                        # High-concurrency vector retrieval latency benchmark
├── test_e2e_integration.py             # Multi-component end-to-end integration test
└── sizing_calculator.py                # Redis memory overhead and capacity calculator
```

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure Python 3.10+ (tested on Python 3.12 & 3.14) is installed. Install all project requirements:

```bash
pip install -r requirements.txt
```

### 2. Run All Tests

Execute the automated test suite across all subsystems:

```bash
python -m pytest tests/
```
*(All 176 unit and integration test cases pass cleanly with 0 failures).*

### 3. Run the Two-Tower Training Demo

Train the Dual-Encoder model on interactions, evaluate convergence, and export normalized embedding artifacts:

```bash
python src/model/train.py
```

### 4. Run Vector Retrieval Benchmarks

Benchmark the feature store and vector retrieval throughput under concurrent loads:

```bash
python benchmark.py
```
*Expected Performance: >30,000 QPS with sub-2.5ms P99 latency.*

### 5. Launch the FastAPI Serving Application

Start the real-time recommendation API service with Uvicorn:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Explore interactive OpenAPI documentation at `http://localhost:8000/docs`.

#### Sample Recommendation Request:
```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "user_101",
       "top_k": 5,
       "context": {
         "season": "Summer",
         "is_weekend": 1,
         "sales_channel_id": 2
       }
     }'
```

---

## 📊 Benchmark & Evaluation Results

* **Retrieval Throughput**: ~34,250 queries/sec
* **Batch Read Latency**: P50 = 1.30 ms | P95 = 1.94 ms | P99 = 2.18 ms (Sub-5ms SLA achieved)
* **Embedding Normalization**: $\mu_{\text{norm}} = 1.0000, \sigma = 0.0000$ (Unit sphere verified)
* **Offline Metrics**: Slicing evaluator reports Recall@10, NDCG@10, and MRR@10 across user frequency cohorts.
