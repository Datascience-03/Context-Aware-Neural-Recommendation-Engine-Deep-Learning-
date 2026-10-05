"""
Member 5 - Day 6-7: API Stress Testing Suite
=============================================
Context-Aware Neural Recommendation Engine

Tests API under concurrent loads including:
  - Baseline latency (single-threaded)
  - Concurrent user load (ThreadPoolExecutor)
  - Ramp-up load patterns
  - Cold-start vs known-user throughput comparison
  - Endpoint-specific stress: /recommend, /health, /metrics
  - Error rate & tail-latency (P50 / P95 / P99) analysis
  - Sustained load soak test (configurable duration)

Usage
-----
Run all stress tests (uses in-process FastAPI TestClient):
    pytest tests/stress_test.py -v

Tune concurrency / request counts via env vars:
    STRESS_WORKERS=20 STRESS_REQUESTS=500 pytest tests/stress_test.py -v
"""

from __future__ import annotations

import os
import sys
import time
import json
import statistics
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

BASE_URL       = os.getenv("STRESS_BASE_URL", "")
WORKERS        = int(os.getenv("STRESS_WORKERS", "10"))
REQUESTS_TOTAL = int(os.getenv("STRESS_REQUESTS", "200"))
SOAK_SECONDS   = int(os.getenv("STRESS_SOAK_SEC", "10"))
TIMEOUT_SEC    = float(os.getenv("STRESS_TIMEOUT", "5.0"))

P50_THRESHOLD_MS  = float(os.getenv("STRESS_P50_MS",  "50"))
P95_THRESHOLD_MS  = float(os.getenv("STRESS_P95_MS",  "150"))
P99_THRESHOLD_MS  = float(os.getenv("STRESS_P99_MS",  "300"))
MAX_ERROR_RATE    = float(os.getenv("STRESS_MAX_ERR",  "0.02"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Synthetic catalog
# ---------------------------------------------------------------------------

_N_USERS = 100
_N_ITEMS = 200
_EMB_DIM = 32

np.random.seed(42)
_U_EMBS = np.random.randn(_N_USERS, _EMB_DIM).astype(np.float32)
_U_EMBS /= np.linalg.norm(_U_EMBS, axis=1, keepdims=True)
_I_EMBS = np.random.randn(_N_ITEMS, _EMB_DIM).astype(np.float32)
_I_EMBS /= np.linalg.norm(_I_EMBS, axis=1, keepdims=True)
_U_IDS  = np.array([f"user_{i}" for i in range(_N_USERS)])
_I_IDS  = np.array([f"item_{i}" for i in range(_N_ITEMS)])


def _build_index():
    from src.retrieval.ann_search import ExactSearchIndex
    idx = ExactSearchIndex(normalize=True)
    idx.fit(_I_IDS, _I_EMBS)
    return idx


def _synthetic_state() -> dict:
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
        "item_index":      _build_index(),
        "user_id_to_idx":  {f"user_{i}": i for i in range(_N_USERS)},
        "eval_metrics":    {"recall@10": 0.72, "ndcg@10": 0.65, "mrr@10": 0.58},
        "startup_time_s":  0.21,
        "redis_client":    None,
    }


import src.api.main as _api_module  # noqa: E402

_api_module._state.update(_synthetic_state())


@asynccontextmanager
async def _noop_lifespan(app):
    yield


_api_module.app.router.lifespan_context = _noop_lifespan

# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

_SEASONS  = ["Winter", "Spring", "Summer", "Fall"]
_CHANNELS = [1, 2]


def _recommend_payload(user_idx: int = 0, top_k: int = 10, cold_start: bool = False) -> dict:
    cid = f"user_COLD_{user_idx}" if cold_start else f"user_{user_idx % _N_USERS}"
    return {
        "customer_id": cid,
        "top_k": top_k,
        "context": {
            "season": _SEASONS[user_idx % len(_SEASONS)],
            "is_weekend": user_idx % 2,
            "sales_channel_id": _CHANNELS[user_idx % len(_CHANNELS)],
        },
    }


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    latency_ms: float
    status_code: int
    success: bool
    error: Optional[str] = None


@dataclass
class LoadTestReport:
    label: str
    total_requests: int
    concurrency: int
    duration_s: float
    results: List[RequestResult] = field(default_factory=list)

    @property
    def successes(self):
        return [r for r in self.results if r.success]

    @property
    def failures(self):
        return [r for r in self.results if not r.success]

    @property
    def error_rate(self) -> float:
        return len(self.failures) / len(self.results) if self.results else 0.0

    @property
    def throughput_rps(self) -> float:
        return len(self.results) / max(self.duration_s, 1e-9)

    def _latencies(self):
        return sorted(r.latency_ms for r in self.successes)

    def percentile(self, p: float) -> float:
        lats = self._latencies()
        if not lats:
            return float("nan")
        idx = max(0, int(len(lats) * p / 100) - 1)
        return lats[min(idx, len(lats) - 1)]

    @property
    def p50(self):
        return self.percentile(50)

    @property
    def p95(self):
        return self.percentile(95)

    @property
    def p99(self):
        return self.percentile(99)

    @property
    def mean_latency(self):
        lats = self._latencies()
        return statistics.mean(lats) if lats else float("nan")

    @property
    def max_latency(self):
        lats = self._latencies()
        return max(lats) if lats else float("nan")

    def summary(self) -> str:
        return (
            f"\n{'─'*60}\n"
            f"  Load Test : {self.label}\n"
            f"{'─'*60}\n"
            f"  Requests    : {self.total_requests}\n"
            f"  Concurrency : {self.concurrency}\n"
            f"  Duration    : {self.duration_s:.2f}s\n"
            f"  Throughput  : {self.throughput_rps:.1f} req/s\n"
            f"  Error Rate  : {self.error_rate * 100:.2f}%\n"
            f"  Mean        : {self.mean_latency:.2f}ms\n"
            f"  P50         : {self.p50:.2f}ms\n"
            f"  P95         : {self.p95:.2f}ms\n"
            f"  P99         : {self.p99:.2f}ms\n"
            f"  Max         : {self.max_latency:.2f}ms\n"
            f"{'─'*60}"
        )


# ---------------------------------------------------------------------------
# Core load runner
# ---------------------------------------------------------------------------

class LoadRunner:
    def __init__(self, client):
        self._client = client
        self._lock   = threading.Lock()

    def _execute(self, fn) -> RequestResult:
        t0 = time.perf_counter()
        try:
            status_code, success = fn()
            ms = (time.perf_counter() - t0) * 1000.0
            return RequestResult(ms, status_code, success)
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000.0
            return RequestResult(ms, 0, False, str(exc))

    def run(self, label, fn_factory, n_requests, concurrency) -> LoadTestReport:
        results = []
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(self._execute, fn_factory(i)): i for i in range(n_requests)}
            for fut in as_completed(futures):
                results.append(fut.result())
        duration_s = time.perf_counter() - t_start
        report = LoadTestReport(label, n_requests, concurrency, duration_s, results)
        print(report.summary())
        return report

    def soak(self, label, fn_factory, duration_s, concurrency) -> LoadTestReport:
        results  = []
        stop     = threading.Event()
        counter  = [0]
        lock     = threading.Lock()
        t_start  = time.perf_counter()

        def _worker():
            while not stop.is_set():
                with lock:
                    idx = counter[0]
                    counter[0] += 1
                res = self._execute(fn_factory(idx))
                with lock:
                    results.append(res)

        threads = [threading.Thread(target=_worker, daemon=True) for _ in range(concurrency)]
        for t in threads:
            t.start()
        time.sleep(duration_s)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        total = time.perf_counter() - t_start
        report = LoadTestReport(label, len(results), concurrency, total, results)
        print(report.summary())
        return report


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    with TestClient(_api_module.app) as c:
        yield c


@pytest.fixture(scope="module")
def runner(client):
    return LoadRunner(client)


# ---------------------------------------------------------------------------
# Request helper factories
# ---------------------------------------------------------------------------

def _make_recommend_fn(client, i, cold_start=False):
    def _fn():
        resp = client.post("/recommend", json=_recommend_payload(i, cold_start=cold_start))
        return resp.status_code, resp.status_code == 200
    return _fn


def _make_health_fn(client):
    def _fn():
        resp = client.get("/health")
        return resp.status_code, resp.status_code == 200
    return _fn


def _make_metrics_fn(client):
    def _fn():
        resp = client.get("/metrics")
        return resp.status_code, resp.status_code == 200
    return _fn


def _make_mixed_fn(client, i):
    mod = i % 10
    if mod < 7:
        return _make_recommend_fn(client, i)
    elif mod < 9:
        return _make_health_fn(client)
    else:
        return _make_metrics_fn(client)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestBaselineLatency:
    """Single-threaded baseline to establish raw latency characteristics."""

    def test_single_recommend_within_p95(self, client):
        payload = _recommend_payload(0, top_k=10)
        t0 = time.perf_counter()
        resp = client.post("/recommend", json=payload)
        ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        assert ms < P95_THRESHOLD_MS, f"Baseline {ms:.1f}ms > P95 threshold {P95_THRESHOLD_MS}ms"

    def test_10_sequential_requests(self, client):
        latencies = []
        for i in range(10):
            t0 = time.perf_counter()
            resp = client.post("/recommend", json=_recommend_payload(i))
            latencies.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status_code == 200
        mean_ms = statistics.mean(latencies)
        print(f"\n  Sequential mean: {mean_ms:.2f}ms")
        assert mean_ms < P95_THRESHOLD_MS

    def test_health_baseline_fast(self, client):
        t0 = time.perf_counter()
        resp = client.get("/health")
        ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        assert ms < 50.0, f"/health took {ms:.1f}ms"

    def test_metrics_baseline_fast(self, client):
        t0 = time.perf_counter()
        resp = client.get("/metrics")
        ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        assert ms < 100.0, f"/metrics took {ms:.1f}ms"


class TestConcurrentRecommendLoad:
    """Concurrent /recommend load at varying concurrency levels."""

    def test_low_concurrency_5_workers(self, runner, client):
        report = runner.run(
            "Low Concurrency (5w x 50r)",
            lambda i: _make_recommend_fn(client, i),
            n_requests=50, concurrency=5,
        )
        assert report.error_rate < MAX_ERROR_RATE
        assert report.p95 < P95_THRESHOLD_MS

    def test_medium_concurrency_10_workers(self, runner, client):
        report = runner.run(
            f"Medium Concurrency ({WORKERS}w x {REQUESTS_TOTAL}r)",
            lambda i: _make_recommend_fn(client, i),
            n_requests=REQUESTS_TOTAL, concurrency=WORKERS,
        )
        assert report.error_rate < MAX_ERROR_RATE

    def test_high_concurrency_20_workers(self, runner, client):
        report = runner.run(
            "High Concurrency (20w x 100r)",
            lambda i: _make_recommend_fn(client, i),
            n_requests=100, concurrency=20,
        )
        assert report.error_rate < 0.05

    def test_concurrent_responses_have_recommendations(self, client):
        results = []
        lock = threading.Lock()

        def _call(i):
            resp = client.post("/recommend", json=_recommend_payload(i))
            with lock:
                results.append(resp)

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for resp in results:
            if resp.status_code == 200:
                data = resp.json()
                assert len(data["recommendations"]) > 0
                assert data["recommendations"][0]["rank"] == 1


class TestColdStartVsKnownUserLoad:
    """Compare throughput and latency: cold-start vs known users."""

    def test_known_user_throughput(self, runner, client):
        report = runner.run(
            "Known Users (8w x 80r)",
            lambda i: _make_recommend_fn(client, i, cold_start=False),
            n_requests=80, concurrency=8,
        )
        assert report.error_rate < MAX_ERROR_RATE

    def test_cold_start_throughput(self, runner, client):
        report = runner.run(
            "Cold-Start Users (8w x 80r)",
            lambda i: _make_recommend_fn(client, i, cold_start=True),
            n_requests=80, concurrency=8,
        )
        assert report.error_rate < MAX_ERROR_RATE

    def test_cold_start_valid_recommendations(self, client):
        results = []
        lock = threading.Lock()

        def _call(i):
            resp = client.post("/recommend", json=_recommend_payload(i, cold_start=True))
            if resp.status_code == 200:
                with lock:
                    results.append(resp.json())

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for data in results:
            assert data["cold_start"] is True
            assert len(data["recommendations"]) > 0
            scores = [r["score"] for r in data["recommendations"]]
            assert scores == sorted(scores, reverse=True)

    def test_latency_comparison(self, client):
        known_lats, cold_lats = [], []
        for i in range(20):
            t0 = time.perf_counter()
            client.post("/recommend", json=_recommend_payload(i, cold_start=False))
            known_lats.append((time.perf_counter() - t0) * 1000.0)
        for i in range(20):
            t0 = time.perf_counter()
            client.post("/recommend", json=_recommend_payload(i, cold_start=True))
            cold_lats.append((time.perf_counter() - t0) * 1000.0)
        mean_k = statistics.mean(known_lats)
        mean_c = statistics.mean(cold_lats)
        print(f"\n  Known={mean_k:.2f}ms | Cold={mean_c:.2f}ms")
        assert mean_c < mean_k * 5 + 20


class TestEndpointSpecificStress:
    """Stress each endpoint in isolation."""

    def test_health_100_concurrent(self, runner, client):
        report = runner.run(
            "/health (20w x 100r)",
            lambda i: _make_health_fn(client),
            n_requests=100, concurrency=20,
        )
        assert report.error_rate == 0.0
        assert report.p99 < 200.0

    def test_metrics_60_concurrent(self, runner, client):
        report = runner.run(
            "/metrics (10w x 60r)",
            lambda i: _make_metrics_fn(client),
            n_requests=60, concurrency=10,
        )
        assert report.error_rate == 0.0

    def test_mixed_endpoint_load(self, runner, client):
        report = runner.run(
            "Mixed Endpoints (15w x 150r)",
            lambda i: _make_mixed_fn(client, i),
            n_requests=150, concurrency=15,
        )
        assert report.error_rate < MAX_ERROR_RATE


class TestRampUpLoad:
    """Gradual ramp-up from 1 to 20 workers."""

    def test_ramp_up_1_to_20(self, client):
        ramp_levels = [1, 2, 5, 10, 15, 20]
        for concurrency in ramp_levels:
            n      = concurrency * 5
            errors = 0
            lats   = []
            lock   = threading.Lock()

            def _call(i):
                nonlocal errors
                t0 = time.perf_counter()
                resp = client.post("/recommend", json=_recommend_payload(i))
                ms   = (time.perf_counter() - t0) * 1000.0
                with lock:
                    lats.append(ms)
                    if resp.status_code != 200:
                        errors += 1

            threads = [threading.Thread(target=_call, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            er = errors / n if n else 0
            print(f"  c={concurrency:2d}: mean={statistics.mean(lats):.1f}ms err={er*100:.1f}%")
            assert er < MAX_ERROR_RATE, f"Error rate {er*100:.2f}% at c={concurrency}"


class TestPayloadVariety:
    """Correctness across varied payloads under concurrency."""

    def test_varied_top_k_concurrent(self, client):
        top_ks  = [1, 5, 10, 20, 50]
        results = {}
        lock    = threading.Lock()

        def _call(top_k, idx):
            payload = {"customer_id": f"user_{idx}", "top_k": top_k,
                       "context": {"season": "Winter", "is_weekend": 0, "sales_channel_id": 1}}
            resp = client.post("/recommend", json=payload)
            with lock:
                results[(top_k, idx)] = resp

        threads = []
        for idx, top_k in enumerate(top_ks * 5):
            threads.append(threading.Thread(target=_call, args=(top_k, idx % _N_USERS)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for (top_k, idx), resp in results.items():
            assert resp.status_code == 200
            assert len(resp.json()["recommendations"]) == min(top_k, _N_ITEMS)

    def test_all_seasons_channels_concurrent(self, client):
        combos = [(s, c) for s in ["Winter", "Spring", "Summer", "Fall"] for c in [1, 2]]
        errors = []
        lock   = threading.Lock()

        def _call(i, season, channel):
            payload = {"customer_id": f"user_{i}", "top_k": 5,
                       "context": {"season": season, "is_weekend": i % 2, "sales_channel_id": channel}}
            resp = client.post("/recommend", json=payload)
            if resp.status_code != 200:
                with lock:
                    errors.append((season, channel, resp.status_code))

        threads = [threading.Thread(target=_call, args=(idx % _N_USERS, s, c))
                   for idx, (s, c) in enumerate(combos * 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_invalid_payloads_return_422(self, client):
        bad = [
            {"customer_id": "u", "top_k": 0},
            {"customer_id": "u", "top_k": 101},
            {"customer_id": "u", "context": {"season": "Monsoon", "is_weekend": 0, "sales_channel_id": 1}},
            {"customer_id": "u", "context": {"season": "Winter", "is_weekend": 5, "sales_channel_id": 1}},
            {"customer_id": "u", "context": {"season": "Winter", "is_weekend": 0, "sales_channel_id": 3}},
        ]
        codes = []
        lock  = threading.Lock()

        def _call(payload):
            resp = client.post("/recommend", json=payload)
            with lock:
                codes.append(resp.status_code)

        threads = [threading.Thread(target=_call, args=(p,)) for p in bad * 4]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for code in codes:
            assert code == 422


class TestDataConsistencyUnderLoad:
    """Recommendation quality and determinism under concurrent load."""

    def test_ordering_deterministic(self, client):
        baseline = client.post("/recommend", json=_recommend_payload(0))
        assert baseline.status_code == 200
        base_ids = [r["article_id"] for r in baseline.json()["recommendations"]]

        results = []
        lock    = threading.Lock()

        def _call():
            resp = client.post("/recommend", json=_recommend_payload(0))
            if resp.status_code == 200:
                with lock:
                    results.append([r["article_id"] for r in resp.json()["recommendations"]])

        threads = [threading.Thread(target=_call) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for ids in results:
            assert ids == base_ids

    def test_no_duplicates_under_load(self, client):
        errors = []
        lock   = threading.Lock()

        def _call(i):
            resp = client.post("/recommend", json=_recommend_payload(i, top_k=20))
            if resp.status_code == 200:
                ids = [r["article_id"] for r in resp.json()["recommendations"]]
                if len(ids) != len(set(ids)):
                    with lock:
                        errors.append(f"user_{i}: duplicates")

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_scores_sorted_descending(self, client):
        errors = []
        lock   = threading.Lock()

        def _call(i):
            resp = client.post("/recommend", json=_recommend_payload(i, top_k=10))
            if resp.status_code == 200:
                scores = [r["score"] for r in resp.json()["recommendations"]]
                if scores != sorted(scores, reverse=True):
                    with lock:
                        errors.append(f"user_{i}")

        threads = [threading.Thread(target=_call, args=(i,)) for i in range(40)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestSLOCompliance:
    """Service Level Objective compliance."""

    def test_p50_slo(self, runner, client):
        report = runner.run(f"SLO P50 ({WORKERS}w x {REQUESTS_TOTAL}r)",
                            lambda i: _make_recommend_fn(client, i),
                            REQUESTS_TOTAL, WORKERS)
        assert report.p50 < P50_THRESHOLD_MS, f"P50 {report.p50:.1f}ms > {P50_THRESHOLD_MS}ms"

    def test_p95_slo(self, runner, client):
        report = runner.run(f"SLO P95 ({WORKERS}w x {REQUESTS_TOTAL}r)",
                            lambda i: _make_recommend_fn(client, i),
                            REQUESTS_TOTAL, WORKERS)
        assert report.p95 < P95_THRESHOLD_MS, f"P95 {report.p95:.1f}ms > {P95_THRESHOLD_MS}ms"

    def test_p99_slo(self, runner, client):
        report = runner.run(f"SLO P99 ({WORKERS}w x {REQUESTS_TOTAL}r)",
                            lambda i: _make_recommend_fn(client, i),
                            REQUESTS_TOTAL, WORKERS)
        assert report.p99 < P99_THRESHOLD_MS, f"P99 {report.p99:.1f}ms > {P99_THRESHOLD_MS}ms"

    def test_error_rate_slo(self, runner, client):
        report = runner.run(f"SLO Error Rate ({WORKERS}w x {REQUESTS_TOTAL}r)",
                            lambda i: _make_recommend_fn(client, i),
                            REQUESTS_TOTAL, WORKERS)
        assert report.error_rate < MAX_ERROR_RATE

    def test_throughput_slo(self, runner, client):
        MIN_RPS = 20.0
        report  = runner.run(f"SLO Throughput ({WORKERS}w x {REQUESTS_TOTAL}r)",
                             lambda i: _make_recommend_fn(client, i),
                             REQUESTS_TOTAL, WORKERS)
        assert report.throughput_rps >= MIN_RPS, \
            f"Throughput {report.throughput_rps:.1f} req/s < SLO {MIN_RPS} req/s"


class TestSoakLoad:
    """Sustained soak test to detect degradation."""

    def test_soak(self, runner, client):
        report = runner.soak(
            f"Soak ({WORKERS}w x {SOAK_SECONDS}s)",
            lambda i: _make_recommend_fn(client, i),
            duration_s=SOAK_SECONDS, concurrency=WORKERS,
        )
        assert report.total_requests > 0
        assert report.error_rate < MAX_ERROR_RATE
        assert report.p99 < P99_THRESHOLD_MS * 2
        print(f"\n  Soak: {report.throughput_rps:.1f} req/s over {SOAK_SECONDS}s")


class TestModelNotReadyUnderLoad:
    """Verify 503 is consistent under concurrent load."""

    def test_503_concurrent(self, client):
        _api_module._state["ready"] = False
        try:
            codes = []
            lock  = threading.Lock()

            def _call(i):
                resp = client.post("/recommend", json=_recommend_payload(i))
                with lock:
                    codes.append(resp.status_code)

            threads = [threading.Thread(target=_call, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for code in codes:
                assert code == 503
        finally:
            _api_module._state["ready"] = True
