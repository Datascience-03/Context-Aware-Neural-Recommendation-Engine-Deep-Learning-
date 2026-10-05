# Context-Aware Neural Recommendation Engine (Deep Learning)

A production-grade, end-to-end Context-Aware Two-Tower Neural Recommendation Engine featuring real-time ANN vector search, dynamic contextual feature engineering, low-latency Redis feature caching, FastAPI microservices, automated Airflow orchestration DAGs, and comprehensive concurrent load stress testing.

---

## 🌟 Architecture Overview

The system implements a modern four-tier recommendation architecture:

```
[ Clients / Frontends ]
          │
          ▼ HTTP REST (JSON)
┌────────────────────────────────────────────────────────┐
│ FastAPI Serving Layer (Port 8000)                      │
│ - /recommend, /health, /metrics, /bulk_query           │
│ - Sub-10ms P50 Latency SLOs                           │
└───────────┬────────────────────────────────┬───────────┘
            │                                │
            ▼                                ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│ User Tower (PyTorch/TF) │      │ Low-Latency Feature     │
│ - Categorical & Context │      │ Store (Redis Cache)     │
│ - Cyclical Time Encodings│     │ - User Embeddings TTL   │
└───────────┬─────────────┘      └─────────────────────────┘
            │ Query Vector (d=64)
            ▼
┌────────────────────────────────────────────────────────┐
│ Real-Time ANN Retrieval Layer (FAISS HNSW / IVF)       │
│ - Million-scale candidate retrieval in <2ms            │
│ - Top-K scoring with cosine/inner-product similarity   │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Features

- **Two-Tower Neural Retrieval Architecture**: Independent User Tower (incorporating user ID, age, context, season, cyclical calendar features, recency/frequency stats) and Item/Candidate Tower.
- **Contextual Feature Engineering**: Temporal sinusoids, seasonal bucketing, user inter-purchase intervals, and dynamic item popularity scoring.
- **Approximate Nearest Neighbors (ANN)**: FAISS-powered HNSW (`M=32`, `ef_search=64`) and IVF (`nlist=100`, `nprobe=10`) indexing for sub-2ms vector retrieval.
- **High-Throughput Async API**: FastAPI endpoints with Pydantic request validation, graceful cold-start fallback, and Prometheus metrics.
- **Automated Airflow Orchestration**:
  - `weekly_full_train`: Full end-to-end data ingestion, neural model retraining, index building, and validation.
  - `daily_incremental_update`: Fast incremental item vector updates and index refreshing.
- **Member 5 API Stress Testing Suite**: 29/29 concurrent load tests across 10 test classes validating strict P50 (<10ms), P95 (<25ms), P99 (<50ms) SLOs.
- **Interactive Architecture Dashboard**: Standalone dark glassmorphism dashboard (`outputs/architecture_diagram.html`) visualizing end-to-end flows.

---

## 📁 Repository Structure

```
├── configs/
│   └── config.yaml                 # Central project configuration
├── src/
│   ├── api/                        # FastAPI recommendation service & endpoints
│   ├── data_processing/            # Data ingestion, cleaning & preprocessing
│   ├── embeddings/                 # Item vector extraction & normalization
│   ├── evaluation/                 # Metrics (Recall@K, MRR@K, NDCG@K, Diversity)
│   ├── feature_engineering/        # Contextual feature extraction & transforms
│   ├── feature_store/              # Redis client & feature caching
│   ├── model/                      # Two-Tower PyTorch/TF neural architecture
│   ├── pipeline/                   # Airflow DAG definitions & orchestration
│   └── retrieval/                  # FAISS HNSW/IVF ANN retrieval engine
├── outputs/
│   └── architecture_diagram.html   # Interactive 7-tab system architecture dashboard
├── tests/
│   ├── stress_test.py              # Member 5 concurrent API load testing suite (29 tests)
│   ├── test_api.py                 # FastAPI endpoint tests
│   ├── test_retrieval.py           # ANN index & search tests
│   ├── test_evaluation.py          # Metric computation tests
│   └── ...                         # 140+ unit and integration tests
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/Context-Aware-Neural-Recommendation-Engine-Deep-Learning.git
   cd Context-Aware-Neural-Recommendation-Engine-Deep-Learning
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Running Tests

### Run Full Test Suite
```bash
pytest tests/
```

### Run Member 5 Concurrent API Stress Tests
```bash
pytest tests/stress_test.py -v
```

### Configurable Stress Test Parameters
```bash
STRESS_WORKERS=20 STRESS_REQUESTS=100 pytest tests/stress_test.py
```

---

## 📊 Stress Test & SLO Compliance

| Metric / Objective | Target SLO | Observed Result | Status |
| :--- | :--- | :--- | :--- |
| **P50 Latency** | < 10.0 ms | **4.92 ms** | ✅ PASS |
| **P95 Latency** | < 25.0 ms | **19.80 ms** | ✅ PASS |
| **P99 Latency** | < 50.0 ms | **24.50 ms** | ✅ PASS |
| **Error Rate** | < 0.1% | **0.00%** | ✅ PASS |
| **Throughput** | > 500 req/s | **820+ req/s** | ✅ PASS |
| **Cold Start Fallback** | 100% Valid Recommendations | **100%** | ✅ PASS |

---

## 👥 Team Work Breakdown

- **Member 1**: Data Pipeline, Cleaning, Ingestion, and Baseline Models.
- **Member 2**: Contextual Feature Engineering, Temporal Encoding, and Aggregations.
- **Member 3**: Two-Tower Neural Network Architecture, Embedding Layers, and Loss Functions.
- **Member 4**: ANN Indexing (FAISS HNSW/IVF), Redis Caching, and Airflow DAG Orchestration.
- **Member 5**: FastAPI Serving, Concurrent API Stress Testing Suite, and System Architecture Diagrams.

---

## 📄 License
MIT License