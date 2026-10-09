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
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.ann_search import ExactSearchIndex, IVFIndex
from src.feature_store.redis_store import RedisUserProfileStore
from src.model.query_tower import QueryTower

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (overridable via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR       = os.getenv("MODEL_DIR",    "data/processed/model")
TOP_K           = int(os.getenv("TOP_K",    "20"))
REDIS_URL       = os.getenv("REDIS_URL",    "")
QUERY_TOWER_WEIGHTS = os.getenv(
    "QUERY_TOWER_WEIGHTS",
    "data/processed/model_export/query_tower.weights.h5",
)

REDIS_FEATURE_URL = os.getenv(
    "REDIS_FEATURE_URL",
    REDIS_URL or "redis://localhost:6379/0",
)

QUERY_TOWER_NUM_USERS = int(os.getenv("QUERY_TOWER_NUM_USERS", "51528"))
QUERY_TOWER_EMBEDDING_DIM = 64

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
    # Redis user feature store + trained QueryTower
    "redis_store":       None,
    "query_tower":       None, 
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
        logger.warning("Cache write skipped: Redis client is unavailable.")
        return

    try:
        client.setex(key, REDIS_TTL_SEC, json.dumps(value))
        logger.info("Recommendation cache saved: %s", key)
    except Exception:
        logger.exception("Failed to save recommendation cache: %s", key)


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
        # Initialize Redis user feature store
    logger.info("Connecting to Redis user feature store: %s", REDIS_FEATURE_URL)

    redis_store = RedisUserProfileStore(
        redis_url=REDIS_FEATURE_URL,
    )

    try:
        redis_store.ping()
        _state["redis_store"] = redis_store
        logger.info("Redis user feature store connected successfully.")
    except Exception as exc:
        redis_store.close()
        raise RuntimeError(
            f"Redis user feature store is unavailable at "
            f"{REDIS_FEATURE_URL}: {exc}"
        ) from exc

    # Load the trained QueryTower used to convert Redis user features
    # into the 64-dimensional query embedding used by retrieval.
    logger.info(
        "Loading QueryTower weights from: %s",
        QUERY_TOWER_WEIGHTS,
    )

    query_tower = QueryTower(
        num_users=QUERY_TOWER_NUM_USERS,
        embedding_dim=QUERY_TOWER_EMBEDDING_DIM,
        user_emb_dim=32,
        context_proj_dim=16,
        hidden_dim=128,
        dropout_rate=0.10,
    )

    # Build the Keras model before loading the weights.
    dummy_inputs = {
        "customer_id_idx": tf.constant([1], dtype=tf.int32),
        "month_sin": tf.constant([0.0], dtype=tf.float32),
        "month_cos": tf.constant([1.0], dtype=tf.float32),
        "day_of_week_sin": tf.constant([0.0], dtype=tf.float32),
        "day_of_week_cos": tf.constant([1.0], dtype=tf.float32),
        "is_weekend": tf.constant([0], dtype=tf.float32),
        "days_since_last_purchase": tf.constant([0.0], dtype=tf.float32),
        "purchase_sequence": tf.constant([0.0], dtype=tf.float32),
    }

    query_tower(dummy_inputs, training=False)
    query_tower.load_weights(QUERY_TOWER_WEIGHTS)

    _state["query_tower"] = query_tower

    logger.info(
        "QueryTower loaded successfully — output dimension=%d.",
        QUERY_TOWER_EMBEDDING_DIM,
    )

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
def _get_query_embedding_from_redis(
    customer_id: str,
) -> Optional[np.ndarray]:
    """Fetch user features from Redis and generate a QueryTower embedding."""

    store = _state["redis_store"]
    tower = _state["query_tower"]

    if store is None or tower is None:
        raise RuntimeError(
            "Redis feature store or QueryTower is not initialized."
        )

    features = store.get_query_features(customer_id)

    if features is None:
        return None

    required = [
        "customer_id_idx",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "days_since_last_purchase",
        "purchase_sequence",
    ]

    missing = [
        name for name in required
        if name not in features
    ]

    if missing:
        raise ValueError(
            f"Missing Redis query features for {customer_id}: {missing}"
        )

    inputs = {
        "customer_id_idx": tf.constant(
            [int(features["customer_id_idx"])],
            dtype=tf.int32,
        ),
        "month_sin": tf.constant(
            [float(features["month_sin"])],
            dtype=tf.float32,
        ),
        "month_cos": tf.constant(
            [float(features["month_cos"])],
            dtype=tf.float32,
        ),
        "day_of_week_sin": tf.constant(
            [float(features["day_of_week_sin"])],
            dtype=tf.float32,
        ),
        "day_of_week_cos": tf.constant(
            [float(features["day_of_week_cos"])],
            dtype=tf.float32,
        ),
        "is_weekend": tf.constant(
            [float(features["is_weekend"])],
            dtype=tf.float32,
        ),
        "days_since_last_purchase": tf.constant(
            [float(features["days_since_last_purchase"])],
            dtype=tf.float32,
        ),
        "purchase_sequence": tf.constant(
            [float(features["purchase_sequence"])],
            dtype=tf.float32,
        ),
    }

    embedding = tower(inputs, training=False).numpy()

    return embedding.astype(np.float32)

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
    Return top-K personalised item recommendations.

    - Fetches user features from Redis.
    - Uses QueryTower to generate a 64-dimensional query embedding.
    - Falls back to the catalog mean embedding for unknown users.
    - Retrieves the top-K items.
    - Optionally serves cached recommendations.
    """
 
    t_start = time.perf_counter()

    cache_key = (
        f"rec:v2:{request.customer_id}:"
        f"{request.context.season}:"
        f"{request.context.is_weekend}:"
        f"{request.top_k}"
    )

    # Cache lookup
    cached = _cache_get(cache_key)

    if cached is not None:
        recs = [
            RecommendationItem(
                rank=i + 1,
                article_id=item["article_id"],
                score=float(item["score"]),
            )
            for i, item in enumerate(cached)
        ]

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return RecommendResponse(
            customer_id=request.customer_id,
            context=request.context,
            recommendations=recs,
            retrieval_time_ms=round(elapsed_ms, 3),
            cache_hit=True,
            cold_start=False,
        )
    # Fetch user features from Redis and generate query embedding
    cold_start = False

    user_emb = _get_query_embedding_from_redis(
        request.customer_id
    )

    if user_emb is None:
        logger.info(
            "Redis profile not found for user '%s'. "
            "Using mean embedding.",
            request.customer_id,
        )

        user_emb = _fallback_user_embedding()
        cold_start = True

    # Retrieve top-K (runs for both normal and cold-start users)
    effective_k = min(request.top_k, _state["num_items"])

    recommendations = _retrieve_top_k(
        user_emb,
        top_k=effective_k,
    )

  

    # Cache result: preserve article IDs and recommendation scores
    _cache_set(
        cache_key,
        [
            {
                "article_id": r.article_id,
                "score": float(r.score),
            }
            for r in recommendations
        ],
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        "Recommend: customer=%s, k=%d, cold_start=%s, latency=%.2fms",
        request.customer_id,
        effective_k,
        cold_start,
        elapsed_ms,
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
