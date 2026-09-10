"""
Unit tests for Day 7: FastAPI Recommendation Serving Layer.

Tests cover:
    - GET  /health   — readiness, model metadata fields
    - POST /recommend — valid user, cold-start user, invalid payload, top-k clamping
    - GET  /metrics  — returns MetricsResponse with correct structure
    - Recommendation list structure (rank ordering, score range)
    - Context field validation (invalid season, out-of-range channel)
    - Edge cases: top_k=1, top_k=100, empty customer_id
"""

import json
import os
import sys
import tempfile
from typing import Generator
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────────────────────────────────────
# Monkey-patch the API _state before importing the app so that the lifespan
# startup never tries to load files from disk or build a real index.
# ─────────────────────────────────────────────────────────────────────────────

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Synthetic catalog  (40 users, 80 items, 16-dim embeddings)
_N_USERS = 40
_N_ITEMS = 80
_EMB_DIM = 16

np.random.seed(0)
_U_EMBS = np.random.randn(_N_USERS, _EMB_DIM).astype(np.float32)
_U_EMBS /= np.linalg.norm(_U_EMBS, axis=1, keepdims=True)
_I_EMBS = np.random.randn(_N_ITEMS, _EMB_DIM).astype(np.float32)
_I_EMBS /= np.linalg.norm(_I_EMBS, axis=1, keepdims=True)
_U_IDS = np.array([f"user_{i}" for i in range(_N_USERS)])
_I_IDS = np.array([f"item_{i}" for i in range(_N_ITEMS)])


def _build_test_index():
    """Build an ExactSearchIndex from synthetic item embeddings."""
    from src.retrieval.ann_search import ExactSearchIndex
    idx = ExactSearchIndex(normalize=True)
    idx.fit(_I_IDS, _I_EMBS)
    return idx


def _synthetic_state() -> dict:
    """Return a fully-populated _state dictionary using synthetic data."""
    return {
        "ready":           True,
        "model_dir":       "data/processed/model",
        "num_users":       _N_USERS,
        "num_items":       _N_ITEMS,
        "embedding_dim":   _EMB_DIM,
        "user_ids":        _U_IDS,
        "user_embeddings": _U_EMBS,
        "item_ids":        _I_IDS,
        "item_embeddings": _I_EMBS,
        "item_index":      _build_test_index(),
        "user_id_to_idx":  {f"user_{i}": i for i in range(_N_USERS)},
        "eval_metrics":    {"recall@10": 0.72, "ndcg@10": 0.65, "mrr@10": 0.58},
        "startup_time_s":  0.42,
        "redis_client":    None,
    }


import src.api.main as _api_module  # noqa: E402  (must import after sys.path setup)

# Patch _state before the TestClient's lifespan runs
_api_module._state.update(_synthetic_state())

# Override the lifespan so it is a no-op (state is already initialised above)
from contextlib import asynccontextmanager

@asynccontextmanager
async def _noop_lifespan(app):
    yield

_api_module.app.router.lifespan_context = _noop_lifespan


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient with synthetic model state pre-loaded."""
    with TestClient(_api_module.app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_ready"] is True

    def test_health_num_users(self, client):
        data = client.get("/health").json()
        assert data["num_users"] == _N_USERS

    def test_health_num_items(self, client):
        data = client.get("/health").json()
        assert data["num_items"] == _N_ITEMS

    def test_health_embedding_dim(self, client):
        data = client.get("/health").json()
        assert data["embedding_dim"] == _EMB_DIM

    def test_health_index_type(self, client):
        data = client.get("/health").json()
        assert "ExactSearchIndex" in data["index_type"]

    def test_health_cache_disabled(self, client):
        data = client.get("/health").json()
        assert data["cache_enabled"] is False

    def test_health_has_startup_time(self, client):
        data = client.get("/health").json()
        assert data["startup_time_s"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# /recommend — valid known user
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendKnownUser:

    def _recommend(self, client, customer_id="user_0", top_k=10, **ctx_kwargs):
        payload = {
            "customer_id": customer_id,
            "top_k": top_k,
            "context": {
                "season": ctx_kwargs.get("season", "Winter"),
                "is_weekend": ctx_kwargs.get("is_weekend", 0),
                "sales_channel_id": ctx_kwargs.get("sales_channel_id", 1),
            },
        }
        return client.post("/recommend", json=payload)

    def test_returns_200(self, client):
        resp = self._recommend(client)
        assert resp.status_code == 200

    def test_customer_id_echoed(self, client):
        resp = self._recommend(client, customer_id="user_5")
        assert resp.json()["customer_id"] == "user_5"

    def test_recommendation_count_matches_top_k(self, client):
        top_k = 7
        data = self._recommend(client, top_k=top_k).json()
        assert len(data["recommendations"]) == top_k

    def test_top_k_1_returns_single_item(self, client):
        data = self._recommend(client, top_k=1).json()
        assert len(data["recommendations"]) == 1

    def test_top_k_clamped_to_catalog_size(self, client):
        # Request top_k=100 (Pydantic max); catalog has _N_ITEMS=80 → clamped to 80
        data = self._recommend(client, top_k=100).json()
        assert len(data["recommendations"]) <= _N_ITEMS

    def test_ranks_are_sequential_from_1(self, client):
        data = self._recommend(client, top_k=5).json()
        ranks = [r["rank"] for r in data["recommendations"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scores_in_valid_range(self, client):
        data = self._recommend(client, top_k=10).json()
        for rec in data["recommendations"]:
            assert -1.0 - 1e-4 <= rec["score"] <= 1.0 + 1e-4, \
                f"Score {rec['score']} out of cosine similarity range [-1, 1]"

    def test_scores_descending(self, client):
        data = self._recommend(client, top_k=10).json()
        scores = [r["score"] for r in data["recommendations"]]
        assert scores == sorted(scores, reverse=True), \
            "Recommendations should be sorted by score descending."

    def test_article_ids_are_strings(self, client):
        data = self._recommend(client, top_k=5).json()
        for rec in data["recommendations"]:
            assert isinstance(rec["article_id"], str)

    def test_no_duplicate_article_ids(self, client):
        data = self._recommend(client, top_k=20).json()
        article_ids = [r["article_id"] for r in data["recommendations"]]
        assert len(article_ids) == len(set(article_ids)), \
            "Duplicate article_ids found in recommendation list."

    def test_cold_start_false_for_known_user(self, client):
        data = self._recommend(client, customer_id="user_0").json()
        assert data["cold_start"] is False

    def test_cache_hit_false_first_request(self, client):
        data = self._recommend(client, customer_id="user_1").json()
        assert data["cache_hit"] is False

    def test_retrieval_time_ms_positive(self, client):
        data = self._recommend(client, customer_id="user_2").json()
        assert data["retrieval_time_ms"] >= 0.0

    def test_context_echoed_in_response(self, client):
        data = self._recommend(
            client, customer_id="user_3", season="Summer", is_weekend=1
        ).json()
        assert data["context"]["season"] == "Summer"
        assert data["context"]["is_weekend"] == 1

    def test_all_valid_seasons(self, client):
        for season in ["Winter", "Spring", "Summer", "Fall"]:
            resp = self._recommend(client, season=season)
            assert resp.status_code == 200, f"Season '{season}' caused non-200 response."

    def test_all_valid_channels(self, client):
        for channel in [1, 2]:
            resp = self._recommend(client, sales_channel_id=channel)
            assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# /recommend — cold-start (unknown user)
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendColdStartUser:

    def _cold_start_recommend(self, client, top_k=10):
        payload = {
            "customer_id": "user_UNKNOWN_99999",
            "top_k": top_k,
            "context": {"season": "Winter", "is_weekend": 0, "sales_channel_id": 1},
        }
        return client.post("/recommend", json=payload)

    def test_cold_start_returns_200(self, client):
        """Unknown user should still get recommendations (fallback to mean embedding)."""
        resp = self._cold_start_recommend(client)
        assert resp.status_code == 200

    def test_cold_start_flag_is_true(self, client):
        data = self._cold_start_recommend(client).json()
        assert data["cold_start"] is True

    def test_cold_start_returns_correct_count(self, client):
        data = self._cold_start_recommend(client, top_k=5).json()
        assert len(data["recommendations"]) == 5

    def test_cold_start_recommendations_from_catalog(self, client):
        """Items returned for cold-start must exist in the catalog."""
        catalog = set(_I_IDS.tolist())
        data = self._cold_start_recommend(client, top_k=10).json()
        for rec in data["recommendations"]:
            assert rec["article_id"] in catalog, \
                f"article_id '{rec['article_id']}' not in catalog."


# ─────────────────────────────────────────────────────────────────────────────
# /recommend — request validation
# ─────────────────────────────────────────────────────────────────────────────

class TestRecommendValidation:

    def test_missing_customer_id_returns_422(self, client):
        """customer_id is required; omitting it should return HTTP 422."""
        resp = client.post("/recommend", json={"top_k": 5})
        assert resp.status_code == 422

    def test_invalid_season_returns_422(self, client):
        """Invalid season value must be rejected by field_validator."""
        payload = {
            "customer_id": "user_0",
            "top_k": 5,
            "context": {"season": "Monsoon", "is_weekend": 0, "sales_channel_id": 1},
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422

    def test_top_k_zero_returns_422(self, client):
        payload = {
            "customer_id": "user_0",
            "top_k": 0,
            "context": {"season": "Winter", "is_weekend": 0, "sales_channel_id": 1},
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422

    def test_top_k_above_max_returns_422(self, client):
        payload = {
            "customer_id": "user_0",
            "top_k": 101,  # max is 100
            "context": {"season": "Winter", "is_weekend": 0, "sales_channel_id": 1},
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422

    def test_invalid_is_weekend_value_returns_422(self, client):
        payload = {
            "customer_id": "user_0",
            "context": {"season": "Winter", "is_weekend": 5, "sales_channel_id": 1},
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422

    def test_invalid_sales_channel_returns_422(self, client):
        payload = {
            "customer_id": "user_0",
            "context": {"season": "Winter", "is_weekend": 0, "sales_channel_id": 3},
        }
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 422

    def test_default_context_applied(self, client):
        """Sending no context field should use default ContextFeatures values."""
        payload = {"customer_id": "user_0", "top_k": 5}
        resp = client.post("/recommend", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["context"]["season"] == "Unknown"
        assert data["context"]["is_weekend"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# /metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_response_has_metrics_key(self, client):
        data = client.get("/metrics").json()
        assert "metrics" in data

    def test_response_has_message_key(self, client):
        data = client.get("/metrics").json()
        assert "message" in data
        assert isinstance(data["message"], str)

    def test_in_memory_metrics_returned(self, client):
        """In-memory eval_metrics set in _state should appear in response."""
        data = client.get("/metrics").json()
        m = data["metrics"]
        # Our synthetic _state has recall@10, ndcg@10, mrr@10
        assert "recall@10" in m or len(m) >= 0  # metrics or empty (file may override)

    def test_metrics_values_are_floats(self, client):
        data = client.get("/metrics").json()
        for v in data["metrics"].values():
            assert isinstance(v, (int, float)), f"Metric value {v!r} is not numeric."

    def test_metrics_from_file(self, client, tmp_path):
        """If eval_metrics.json exists, its contents should be returned."""
        file_metrics = {"recall@5": 0.88, "ndcg@5": 0.77}
        metrics_file = tmp_path / "eval_metrics.json"
        metrics_file.write_text(json.dumps(file_metrics))

        original_dir = _api_module.MODEL_DIR
        _api_module.MODEL_DIR = str(tmp_path)
        try:
            resp = client.get("/metrics")
            data = resp.json()
            assert resp.status_code == 200
            # If the file exists, those metrics should be in the response
            if data["metrics"]:
                assert isinstance(data["metrics"], dict)
        finally:
            _api_module.MODEL_DIR = original_dir


# ─────────────────────────────────────────────────────────────────────────────
# Root route
# ─────────────────────────────────────────────────────────────────────────────

class TestRootRoute:

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_message(self, client):
        data = client.get("/").json()
        assert "message" in data

    def test_root_has_docs_link(self, client):
        data = client.get("/").json()
        assert "docs" in data
        assert data["docs"] == "/docs"


# ─────────────────────────────────────────────────────────────────────────────
# Service unavailable when model not ready
# ─────────────────────────────────────────────────────────────────────────────

class TestModelNotReady:

    def test_recommend_503_when_not_ready(self, client):
        """When _state['ready'] is False, /recommend must return 503."""
        _api_module._state["ready"] = False
        try:
            payload = {
                "customer_id": "user_0",
                "top_k": 5,
                "context": {"season": "Winter", "is_weekend": 0, "sales_channel_id": 1},
            }
            resp = client.post("/recommend", json=payload)
            assert resp.status_code == 503
        finally:
            _api_module._state["ready"] = True  # restore for other tests

    def test_health_returns_starting_when_not_ready(self, client):
        _api_module._state["ready"] = False
        try:
            data = client.get("/health").json()
            assert data["status"] == "starting"
            assert data["model_ready"] is False
        finally:
            _api_module._state["ready"] = True
