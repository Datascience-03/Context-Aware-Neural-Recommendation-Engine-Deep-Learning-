"""
Day 7: FastAPI Serving Layer for the Context-Aware Neural Recommendation Engine.

Endpoints:
    POST /recommend  — Return top-K item recommendations for a user + context
    GET  /health     — Model and index health status
    GET  /metrics    — Latest aggregated retrieval evaluation metrics

The server loads user/item embedding artefacts on startup and builds an
ExactSearchIndex (with optional IVFIndex for high-load paths). Recommendations
are computed via fast dot-product retrieval against the item embedding index.

Optional Redis caching:  if a REDIS_URL environment variable is set and Redis is
reachable, recommendation results are cached with a configurable TTL to avoid
redundant embedding lookups for repeated requests.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.ann_search import ExactSearchIndex, IVFIndex

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (overridable via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR       = os.getenv("MODEL_DIR",    "data/processed/model")
TOP_K           = int(os.getenv("TOP_K",    "20"))
REDIS_URL       = os.getenv("REDIS_URL",    "")
REDIS_TTL_SEC   = int(os.getenv("REDIS_TTL", "300"))
IVF_NLIST       = int(os.getenv("IVF_NLIST",  "32"))
IVF_NPROBE      = int(os.getenv("IVF_NPROBE",  "4"))
USE_IVF         = os.getenv("USE_IVF", "0").lower() in {"1", "true", "yes"}

# ─────────────────────────────────────────────────────────────────────────────
# Global state (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────

_state: Dict[str, Any] = {
    "ready":             False,
    "model_dir":         MODEL_DIR,
    "num_users":         0,
    "num_items":         0,
    "embedding_dim":     0,
    "user_ids":          None,   # np.ndarray[str], shape (N,)
    "user_embeddings":   None,   # np.ndarray[float32], shape (N, D)
    "item_ids":          None,   # np.ndarray[str], shape (M,)
    "item_embeddings":   None,   # np.ndarray[float32], shape (M, D)
    "item_index":        None,   # ExactSearchIndex | IVFIndex
    "user_id_to_idx":   {},
    "eval_metrics":      {},
    "startup_time_s":    0.0,
    "redis_client":      None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Optional Redis helper
# ─────────────────────────────────────────────────────────────────────────────

def _init_redis() -> Optional[Any]:
    """Attempt to connect to Redis. Returns client or None on failure."""
    if not REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        client.ping()
        logger.info("Redis cache connected at %s (TTL=%ds).", REDIS_URL, REDIS_TTL_SEC)
        return client
    except Exception as exc:
        logger.warning("Redis unavailable (%s) — caching disabled.", exc)
        return None


def _cache_get(key: str) -> Optional[List[str]]:
    client = _state["redis_client"]
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def _cache_set(key: str, value: List[str]) -> None:
    client = _state["redis_client"]
    if client is None:
        return
    try:
        client.setex(key, REDIS_TTL_SEC, json.dumps(value))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts and build the retrieval index on startup."""
    t_start = time.perf_counter()
    logger.info("Loading recommendation model artefacts from: %s", MODEL_DIR)

    model_path = Path(MODEL_DIR)
    artefact_keys = ["user_ids", "user_embeddings", "item_ids", "item_embeddings"]
    artefacts_exist = all((model_path / f"{k}.npy").exists() for k in artefact_keys)

    if artefacts_exist:
        _state["user_ids"]        = np.load(str(model_path / "user_ids.npy"),        allow_pickle=True)
        _state["user_embeddings"] = np.load(str(model_path / "user_embeddings.npy"))
        _state["item_ids"]        = np.load(str(model_path / "item_ids.npy"),        allow_pickle=True)
        _state["item_embeddings"] = np.load(str(model_path / "item_embeddings.npy"))
        logger.info(
            "Embeddings loaded — users: %s, items: %s",
            _state["user_embeddings"].shape,
            _state["item_embeddings"].shape,
        )
    else:
        logger.warning(
            "Model artefacts not found in %s. "
            "Generating synthetic demo embeddings (run src/model/train.py to train a real model).",
            MODEL_DIR,
        )
        np.random.seed(0)
        n_u, n_i, d = 200, 500, 64
        u_embs = np.random.randn(n_u, d).astype(np.float32)
        u_embs /= np.linalg.norm(u_embs, axis=1, keepdims=True)
        i_embs = np.random.randn(n_i, d).astype(np.float32)
        i_embs /= np.linalg.norm(i_embs, axis=1, keepdims=True)
        _state["user_ids"]        = np.array([f"user_{i}" for i in range(n_u)])
        _state["user_embeddings"] = u_embs
        _state["item_ids"]        = np.array([f"item_{i}" for i in range(n_i)])
        _state["item_embeddings"] = i_embs

    # Pre-compute lookup map: user_id_string → embedding row index
    _state["user_id_to_idx"] = {
        str(uid): idx for idx, uid in enumerate(_state["user_ids"])
    }

    # Build retrieval index
    if USE_IVF and len(_state["item_ids"]) >= IVF_NLIST * 2:
        logger.info("Building IVFIndex (nlist=%d, nprobe=%d)...", IVF_NLIST, IVF_NPROBE)
        idx = IVFIndex(nlist=IVF_NLIST, nprobe=IVF_NPROBE, normalize=True, random_state=42)
    else:
        logger.info("Building ExactSearchIndex...")
        idx = ExactSearchIndex(normalize=True)

    idx.fit(_state["item_ids"], _state["item_embeddings"])
    _state["item_index"]    = idx
    _state["num_users"]     = len(_state["user_ids"])
    _state["num_items"]     = len(_state["item_ids"])
    _state["embedding_dim"] = _state["item_embeddings"].shape[1]

    # Optional Redis cache
    _state["redis_client"] = _init_redis()

    _state["startup_time_s"] = time.perf_counter() - t_start
    _state["ready"]          = True

    logger.info(
        "API ready in %.2fs — %d users, %d items, dim=%d, index=%s.",
        _state["startup_time_s"],
        _state["num_users"],
        _state["num_items"],
        _state["embedding_dim"],
        type(idx).__name__,
    )

    yield  # ── server is running ──

    logger.info("Shutting down — releasing model resources.")
    _state["ready"] = False


# ─────────────────────────────────────────────────────────────────────────────
# App definition
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    description=(
        "Two-Tower deep learning recommendation API. "
        "Provides top-K personalised item recommendations via fast vector retrieval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic request / response schemas
# ─────────────────────────────────────────────────────────────────────────────

class ContextFeatures(BaseModel):
    """User context features for the recommendation request."""
    season: str = Field(
        default="Unknown",
        description="Season name: 'Winter', 'Spring', 'Summer', or 'Fall'.",
        examples=["Winter"],
    )
    is_weekend: int = Field(
        default=0,
        ge=0, le=1,
        description="1 if the request is on a weekend, else 0.",
    )
    sales_channel_id: int = Field(
        default=1,
        ge=1, le=2,
        description="Sales channel: 1 = in-store, 2 = online.",
    )

    @field_validator("season")
    @classmethod
    def validate_season(cls, v: str) -> str:
        allowed = {"Winter", "Spring", "Summer", "Fall", "Unknown"}
        if v not in allowed:
            raise ValueError(f"season must be one of {allowed}, got '{v}'")
        return v


class RecommendRequest(BaseModel):
    """Request payload for the /recommend endpoint."""
    customer_id: str = Field(
        ...,
        description="Unique customer identifier (must match training vocabulary).",
        examples=["user_42"],
    )
    context: ContextFeatures = Field(
        default_factory=ContextFeatures,
        description="Optional contextual signals for this recommendation session.",
    )
    top_k: int = Field(
        default=10,
        ge=1, le=100,
        description="Number of top recommendations to return.",
    )
    exclude_seen: bool = Field(
        default=False,
        description="If True, attempt to exclude previously interacted items (requires history store).",
    )


class RecommendationItem(BaseModel):
    """Single ranked recommendation result."""
    rank: int
    article_id: str
    score: float = Field(description="Cosine similarity score in [-1, 1].")


class RecommendResponse(BaseModel):
    """Response payload for the /recommend endpoint."""
    customer_id: str
    context: ContextFeatures
    recommendations: List[RecommendationItem]
    retrieval_time_ms: float
    cache_hit: bool = False
    cold_start: bool = False


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str
    model_ready: bool
    num_users: int
    num_items: int
    embedding_dim: int
    index_type: str
    startup_time_s: float
    cache_enabled: bool


class MetricsResponse(BaseModel):
    """Response for /metrics endpoint."""
    metrics: Dict[str, float]
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Core recommendation logic
# ─────────────────────────────────────────────────────────────────────────────

def _get_user_embedding(customer_id: str) -> Optional[np.ndarray]:
    """
    Retrieve the stored embedding for a known user.
    Returns None if the user is unknown (cold-start).

    Args:
        customer_id: User identifier string.

    Returns:
        np.ndarray of shape (1, D) or None.
    """
    idx = _state["user_id_to_idx"].get(str(customer_id))
    if idx is None:
        return None
    return _state["user_embeddings"][idx : idx + 1]


def _fallback_user_embedding() -> np.ndarray:
    """Return the mean of all user embeddings as a cold-start fallback."""
    mean_emb = np.mean(_state["user_embeddings"], axis=0, keepdims=True).astype(np.float32)
    norm = np.linalg.norm(mean_emb, axis=1, keepdims=True)
    return mean_emb / np.maximum(norm, 1e-12)


def _retrieve_top_k(
    user_emb: np.ndarray,
    top_k: int,
) -> List[RecommendationItem]:
    """
    Run top-K retrieval from the item index for a given user embedding.

    Args:
        user_emb: np.ndarray of shape (1, D).
        top_k: Number of candidates to retrieve.

    Returns:
        List of RecommendationItem objects in ranked order.
    """
    index = _state["item_index"]
    retrieved_ids, scores = index.search(user_emb, k=top_k)

    results = []
    for rank_i, (item_id, score) in enumerate(
        zip(retrieved_ids[0], scores[0]), start=1
    ):
        if item_id is not None:
            results.append(RecommendationItem(
                rank=rank_i,
                article_id=str(item_id),
                score=round(float(score), 6),
            ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get personalised item recommendations for a customer.",
    response_description="Ranked list of recommended article IDs.",
    tags=["Recommendations"],
)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Return top-K personalised item recommendations for a customer with optional context.

    - Looks up the pre-computed user embedding for `customer_id`.
    - Falls back to the catalog mean embedding for unknown (cold-start) users.
    - Performs fast dot-product retrieval against the item embedding index.
    - Optionally serves from Redis cache to reduce latency on repeated requests.
    """
    if not _state["ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready yet. Please retry in a moment.",
        )

    t_start = time.perf_counter()
    cache_key = f"rec:{request.customer_id}:{request.context.season}:{request.context.is_weekend}:{request.top_k}"

    # ── Cache lookup ──────────────────────────────────────────────────────────
    cached = _cache_get(cache_key)
    if cached is not None:
        recs = [
            RecommendationItem(rank=i + 1, article_id=item_id, score=0.0)
            for i, item_id in enumerate(cached)
        ]
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return RecommendResponse(
            customer_id=request.customer_id,
            context=request.context,
            recommendations=recs,
            retrieval_time_ms=round(elapsed_ms, 3),
            cache_hit=True,
        )

    # ── User embedding lookup (cold-start handling) ───────────────────────────
    cold_start = False
    user_emb = _get_user_embedding(request.customer_id)
    if user_emb is None:
        logger.info("Cold-start user: '%s'. Using mean embedding.", request.customer_id)
        user_emb = _fallback_user_embedding()
        cold_start = True

    # ── Retrieve top-K ────────────────────────────────────────────────────────
    effective_k = min(request.top_k, _state["num_items"])
    recommendations = _retrieve_top_k(user_emb, top_k=effective_k)

    # ── Cache result ──────────────────────────────────────────────────────────
    _cache_set(cache_key, [r.article_id for r in recommendations])

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        "Recommend: customer=%s, k=%d, cold_start=%s, latency=%.2fms",
        request.customer_id, effective_k, cold_start, elapsed_ms,
    )

    return RecommendResponse(
        customer_id=request.customer_id,
        context=request.context,
        recommendations=recommendations,
        retrieval_time_ms=round(elapsed_ms, 3),
        cache_hit=False,
        cold_start=cold_start,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Model and index health check.",
    tags=["Operations"],
)
async def health() -> HealthResponse:
    """
    Return service health status and model metadata.
    """
    index = _state.get("item_index")
    index_type = type(index).__name__ if index is not None else "None"
    return HealthResponse(
        status="ok" if _state["ready"] else "starting",
        model_ready=_state["ready"],
        num_users=_state["num_users"],
        num_items=_state["num_items"],
        embedding_dim=_state["embedding_dim"],
        index_type=index_type,
        startup_time_s=round(_state["startup_time_s"], 3),
        cache_enabled=_state["redis_client"] is not None,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Latest retrieval evaluation metrics.",
    tags=["Operations"],
)
async def metrics() -> MetricsResponse:
    """
    Return the most recent Recall@K / NDCG@K / MRR@K metrics from the evaluation pipeline.
    These are populated after running `src/model/evaluate.py` and persisted to
    `data/processed/model/eval_metrics.json`.
    """
    metrics_path = Path(MODEL_DIR) / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            eval_metrics = json.load(f)
        msg = "Metrics loaded from last evaluation run."
    elif _state["eval_metrics"]:
        eval_metrics = _state["eval_metrics"]
        msg = "In-memory metrics (no file found)."
    else:
        eval_metrics = {}
        msg = (
            "No evaluation metrics available. "
            "Run `python src/model/evaluate.py` to generate them."
        )

    return MetricsResponse(metrics=eval_metrics, message=msg)


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        {"message": "Context-Aware Neural Recommendation Engine API", "docs": "/docs"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for uvicorn
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
