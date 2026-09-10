"""
Day 4: Approximate Nearest Neighbor (ANN) Retrieval & Latency/Recall Trade-off Evaluation.
Provides VectorSearchIndex interface, ExactSearchIndex (brute-force baseline),
IVFIndex (Inverted File Index with K-Means Voronoi partitioning),
and ANNEvaluator for benchmarking Recall Retention vs. Latency / Throughput (QPS).
"""

import os
import sys
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

# Append workspace root to path for standalone execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logger = logging.getLogger(__name__)

# Optional Faiss dependency check
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize rows of a 2D float array.

    Args:
        vectors: 2D numpy array of shape (N, D).
        eps: Small epsilon to prevent division by zero.

    Returns:
        L2-normalized 2D numpy array of shape (N, D).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (vectors / norms).astype(np.float32)


class VectorSearchIndex(ABC):
    """
    Abstract Base Class for Vector Retrieval Indices.
    Defines the contract for index construction (fit/add) and top-K search.
    """

    def __init__(self, normalize: bool = True, metric: str = "cosine"):
        """
        Initialize base vector search index.

        Args:
            normalize: If True, vectors are L2-normalized (cosine similarity).
                       If False, raw inner product is used.
            metric: Distance/similarity metric name ("cosine" or "inner_product").
        """
        self.normalize = normalize
        self.metric = metric
        self.item_ids: np.ndarray = np.array([])
        self.dim: int = 0
        self.is_fitted: bool = False

    @property
    def num_items(self) -> int:
        """Total number of items indexed."""
        return len(self.item_ids)

    @abstractmethod
    def fit(self, item_ids: Sequence[Any], embeddings: np.ndarray) -> "VectorSearchIndex":
        """
        Build and populate index with item identifiers and embedding vectors.

        Args:
            item_ids: Sequence of N item IDs.
            embeddings: 2D numpy array of shape (N, D).

        Returns:
            self
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-K nearest items for given query embeddings.

        Args:
            query_embeddings: 2D numpy array of shape (Q, D).
            k: Number of nearest items to retrieve per query.
            **kwargs: Index-specific search parameters (e.g. nprobe).

        Returns:
            Tuple of (retrieved_item_ids, scores):
                - retrieved_item_ids: shape (Q, K) of item identifiers.
                - scores: shape (Q, K) of similarity scores.
        """
        pass


class ExactSearchIndex(VectorSearchIndex):
    """
    Exhaustive brute-force vector search index.
    Serves as the ground-truth baseline for exact top-K candidate retrieval.
    """

    def __init__(self, normalize: bool = True, metric: str = "cosine"):
        super().__init__(normalize=normalize, metric=metric)
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def fit(self, item_ids: Sequence[Any], embeddings: np.ndarray) -> "ExactSearchIndex":
        """
        Store item identifiers and embeddings.

        Args:
            item_ids: Sequence of N item IDs.
            embeddings: 2D numpy array of shape (N, D).

        Returns:
            self
        """
        embeddings_arr = np.asarray(embeddings, dtype=np.float32)
        if embeddings_arr.ndim != 2:
            raise ValueError(f"Embeddings must be 2-dimensional (N, D), got shape {embeddings_arr.shape}")
        if len(item_ids) != embeddings_arr.shape[0]:
            raise ValueError(
                f"Mismatch: len(item_ids)={len(item_ids)} != embeddings.shape[0]={embeddings_arr.shape[0]}"
            )

        self.item_ids = np.array(item_ids)
        self.dim = embeddings_arr.shape[1]

        if self.normalize:
            self.embeddings = _normalize_vectors(embeddings_arr)
        else:
            self.embeddings = embeddings_arr

        self.is_fitted = True
        return self

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform exhaustive brute-force top-K search via matrix multiplication.

        Args:
            query_embeddings: 2D array of shape (Q, D).
            k: Top-K items to retrieve per query.

        Returns:
            Tuple of (retrieved_ids, scores), each of shape (Q, min(k, N)).
        """
        if not self.is_fitted or self.num_items == 0:
            raise RuntimeError("Index must be fitted with items before search.")

        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension {queries.shape[1]} does not match index dimension {self.dim}"
            )

        if self.normalize:
            queries = _normalize_vectors(queries)

        num_queries = queries.shape[0]
        eff_k = min(k, self.num_items)
        if eff_k <= 0:
            return np.empty((num_queries, 0)), np.empty((num_queries, 0), dtype=np.float32)

        # Batch inner product computation: (Q, D) @ (D, N) -> (Q, N)
        similarity_matrix = np.matmul(queries, self.embeddings.T)

        if eff_k >= self.num_items:
            # Full sort when k equals or exceeds total items
            sorted_indices = np.argsort(-similarity_matrix, axis=1)
            top_ids = self.item_ids[sorted_indices]
            top_scores = np.take_along_axis(similarity_matrix, sorted_indices, axis=1)
            return top_ids, top_scores

        # Efficient top-K selection using argpartition
        part_indices = np.argpartition(-similarity_matrix, eff_k, axis=1)[:, :eff_k]
        part_scores = np.take_along_axis(similarity_matrix, part_indices, axis=1)
        sort_order = np.argsort(-part_scores, axis=1)

        final_indices = np.take_along_axis(part_indices, sort_order, axis=1)
        top_ids = self.item_ids[final_indices]
        top_scores = np.take_along_axis(similarity_matrix, final_indices, axis=1)

        return top_ids, top_scores


class IVFIndex(VectorSearchIndex):
    """
    Inverted File (IVF) Approximate Nearest Neighbor (ANN) Index.
    Partitions the embedding space into Voronoi cells using spherical K-Means clustering.
    Provides sub-linear search time by probing only the nearest `nprobe` centroids.
    """

    def __init__(
        self,
        nlist: int = 32,
        nprobe: int = 1,
        normalize: bool = True,
        max_iter: int = 25,
        metric: str = "cosine",
        random_state: int = 42,
        use_faiss: bool = False,
    ):
        """
        Initialize IVF index.

        Args:
            nlist: Number of Voronoi partitions (centroids/clusters).
            nprobe: Default number of centroids to probe during search (1 <= nprobe <= nlist).
            normalize: If True, normalize vectors to unit hypersphere (cosine similarity).
            max_iter: Maximum K-Means iterations during centroid training.
            metric: Similarity metric ("cosine" or "inner_product").
            random_state: Random seed for centroid initialization.
            use_faiss: If True and faiss is available, use Faiss IndexIVFFlat engine.
                       Defaults to False (pure NumPy engine).
        """
        super().__init__(normalize=normalize, metric=metric)
        self.nlist = max(1, nlist)
        self.nprobe = max(1, nprobe)
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_faiss = use_faiss and HAS_FAISS

        self.centroids: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.inverted_lists: Dict[int, np.ndarray] = {}
        self.embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.faiss_index: Optional[Any] = None

    def fit(self, item_ids: Sequence[Any], embeddings: np.ndarray) -> "IVFIndex":
        """
        Cluster item embeddings using K-Means and construct inverted file lists.

        Args:
            item_ids: Sequence of N item IDs.
            embeddings: 2D numpy array of shape (N, D).

        Returns:
            self
        """
        embeddings_arr = np.asarray(embeddings, dtype=np.float32)
        if embeddings_arr.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings_arr.shape}")
        if len(item_ids) != embeddings_arr.shape[0]:
            raise ValueError(
                f"Mismatch: len(item_ids)={len(item_ids)} != embeddings.shape[0]={embeddings_arr.shape[0]}"
            )

        n_samples, self.dim = embeddings_arr.shape
        self.item_ids = np.array(item_ids)

        if self.normalize:
            self.embeddings = _normalize_vectors(embeddings_arr)
        else:
            self.embeddings = embeddings_arr

        # Adjust nlist if items count is smaller than requested clusters
        self.nlist = min(self.nlist, max(1, n_samples))

        if self.use_faiss and HAS_FAISS:
            self._fit_faiss(self.embeddings)
        else:
            self._fit_numpy(self.embeddings)

        self.is_fitted = True
        return self

    def _fit_faiss(self, embeddings: np.ndarray) -> None:
        """Fit index using Faiss library if enabled."""
        quantizer = faiss.IndexFlatIP(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = self.nprobe
        self.faiss_index = index

    def _fit_numpy(self, embeddings: np.ndarray) -> None:
        """
        Fit spherical K-Means centroids and construct inverted lists natively in NumPy.
        """
        n_samples = embeddings.shape[0]
        rng = np.random.RandomState(self.random_state)

        # 1. K-Means Initialization (Random Distinct Centroids)
        init_indices = rng.choice(n_samples, size=self.nlist, replace=False)
        centroids = embeddings[init_indices].copy()
        if self.normalize:
            centroids = _normalize_vectors(centroids)

        # 2. Iterative Spherical K-Means Centroid Updates
        labels = np.zeros(n_samples, dtype=np.int32)
        for _ in range(self.max_iter):
            # Compute similarity to current centroids: (N, nlist)
            similarities = np.matmul(embeddings, centroids.T)
            new_labels = np.argmax(similarities, axis=1)

            # Check convergence
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Update centroids to cluster means
            for c in range(self.nlist):
                cluster_mask = (labels == c)
                if np.any(cluster_mask):
                    cluster_mean = np.mean(embeddings[cluster_mask], axis=0)
                    centroids[c] = cluster_mean
                else:
                    # Reinitialize empty cluster with a random sample
                    rand_idx = rng.randint(0, n_samples)
                    centroids[c] = embeddings[rand_idx]

            if self.normalize:
                centroids = _normalize_vectors(centroids)

        self.centroids = centroids

        # 3. Construct Inverted Lists (Cluster ID -> Item Indices)
        self.inverted_lists = {}
        for c in range(self.nlist):
            item_indices = np.where(labels == c)[0]
            self.inverted_lists[c] = item_indices

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-K candidates using IVF multi-cell probing.

        Args:
            query_embeddings: 2D array of shape (Q, D).
            k: Number of items to retrieve per query.
            nprobe: Number of nearest centroids to probe. If None, uses self.nprobe.

        Returns:
            Tuple of (retrieved_item_ids, scores), each of shape (Q, eff_k).
        """
        if not self.is_fitted or self.num_items == 0:
            raise RuntimeError("IVFIndex must be fitted before search.")

        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self.dim:
            raise ValueError(
                f"Query dimension {queries.shape[1]} does not match index dimension {self.dim}"
            )

        if self.normalize:
            queries = _normalize_vectors(queries)

        probe_count = self.nprobe if nprobe is None else nprobe
        probe_count = min(max(1, probe_count), self.nlist)

        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.nprobe = probe_count
            eff_k = min(k, self.num_items)
            scores, indices = self.faiss_index.search(queries, eff_k)
            retrieved_ids = np.empty_like(indices, dtype=object)
            valid_mask = (indices >= 0) & (indices < self.num_items)
            retrieved_ids[valid_mask] = self.item_ids[indices[valid_mask]]
            retrieved_ids[~valid_mask] = None
            return retrieved_ids, scores

        return self._search_numpy(queries, k=k, nprobe=probe_count)

    def _search_numpy(
        self,
        queries: np.ndarray,
        k: int,
        nprobe: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute IVF search natively using NumPy Voronoi probing.
        """
        num_queries = queries.shape[0]
        eff_k = min(k, self.num_items)
        if eff_k <= 0:
            return np.empty((num_queries, 0)), np.empty((num_queries, 0), dtype=np.float32)

        # 1. Similarity to centroids: (Q, nlist)
        centroid_sims = np.matmul(queries, self.centroids.T)

        out_ids = []
        out_scores = []

        for q_idx in range(num_queries):
            # Select top nprobe centroids for this query
            q_vec = queries[q_idx]
            c_sims = centroid_sims[q_idx]

            if nprobe >= self.nlist:
                probed_centroids = np.arange(self.nlist)
            else:
                probed_centroids = np.argpartition(-c_sims, nprobe)[:nprobe]

            # Gather candidate item indices across probed cells
            cell_indices = [
                self.inverted_lists[c]
                for c in probed_centroids
                if c in self.inverted_lists and len(self.inverted_lists[c]) > 0
            ]

            if len(cell_indices) == 0:
                # Fallback: if probed cells are empty, search all items
                cand_indices = np.arange(self.num_items)
            else:
                cand_indices = np.concatenate(cell_indices)

            # Compute similarities with candidate embeddings
            cand_embeddings = self.embeddings[cand_indices]
            cand_scores = np.matmul(cand_embeddings, q_vec)

            # Retrieve top-K within candidates
            cand_len = len(cand_indices)
            if cand_len <= eff_k:
                sort_order = np.argsort(-cand_scores)
                chosen_cand_idx = cand_indices[sort_order]
                chosen_scores = cand_scores[sort_order]

                # If fewer candidates than eff_k, pad with fallback if necessary
                if len(chosen_cand_idx) < eff_k:
                    pad_len = eff_k - len(chosen_cand_idx)
                    pad_ids = [None] * pad_len
                    pad_scores = np.full(pad_len, -np.inf, dtype=np.float32)
                    q_ids = np.concatenate([self.item_ids[chosen_cand_idx], pad_ids])
                    q_sc = np.concatenate([chosen_scores, pad_scores])
                else:
                    q_ids = self.item_ids[chosen_cand_idx]
                    q_sc = chosen_scores
            else:
                part = np.argpartition(-cand_scores, eff_k)[:eff_k]
                part_scores = cand_scores[part]
                sort_order = np.argsort(-part_scores)

                chosen_cand_idx = cand_indices[part[sort_order]]
                q_ids = self.item_ids[chosen_cand_idx]
                q_sc = part_scores[sort_order]

            out_ids.append(q_ids)
            out_scores.append(q_sc)

        return np.array(out_ids, dtype=object), np.array(out_scores, dtype=np.float32)


class ANNEvaluator:
    """
    Evaluation & Benchmarking Harness for Approximate Nearest Neighbor (ANN) Retrieval.
    Measures Recall Retention Rate, Latency (Mean, P50, P95, P99), Throughput (QPS),
    and Speedup vs. Exact Ground-Truth Search across hyperparameter sweeps.
    """

    @staticmethod
    def recall_retention_at_k(
        exact_retrieved: np.ndarray,
        ann_retrieved: np.ndarray,
        k: Optional[int] = None,
    ) -> float:
        """
        Calculate average Recall Retention Rate @ K between Exact and ANN retrieval.
        Recall Retention is the fraction of exact top-K items preserved in the ANN top-K.

        Args:
            exact_retrieved: Ground-truth item IDs of shape (Q, K_exact).
            ann_retrieved: ANN retrieved item IDs of shape (Q, K_ann).
            k: Rank cutoff K. If None, uses min(exact.shape[1], ann.shape[1]).

        Returns:
            Mean Recall Retention Rate in [0.0, 1.0].
        """
        num_queries = len(exact_retrieved)
        if num_queries == 0 or len(ann_retrieved) == 0:
            return 0.0

        cutoff = k if k is not None else min(exact_retrieved.shape[1], ann_retrieved.shape[1])
        if cutoff <= 0:
            return 0.0

        retention_scores = []
        for q in range(num_queries):
            exact_set = set(exact_retrieved[q][:cutoff])
            # Remove None padding if present
            exact_set.discard(None)
            if len(exact_set) == 0:
                continue

            ann_set = set(ann_retrieved[q][:cutoff])
            ann_set.discard(None)

            overlap = len(exact_set.intersection(ann_set))
            retention_scores.append(overlap / len(exact_set))

        return float(np.mean(retention_scores)) if len(retention_scores) > 0 else 0.0

    @staticmethod
    def top_1_accuracy(
        exact_retrieved: np.ndarray,
        ann_retrieved: np.ndarray,
    ) -> float:
        """
        Calculate fraction of queries where ANN top-1 item matches Exact top-1 item.

        Args:
            exact_retrieved: Ground-truth item IDs of shape (Q, >=1).
            ann_retrieved: ANN retrieved item IDs of shape (Q, >=1).

        Returns:
            Top-1 accuracy score in [0.0, 1.0].
        """
        num_queries = min(len(exact_retrieved), len(ann_retrieved))
        if num_queries == 0:
            return 0.0

        matches = sum(
            1 for q in range(num_queries)
            if exact_retrieved[q][0] == ann_retrieved[q][0] and exact_retrieved[q][0] is not None
        )
        return float(matches / num_queries)

    @classmethod
    def benchmark_query_latency(
        cls,
        index: VectorSearchIndex,
        queries: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Profile retrieval latency and throughput over multiple timed iterations.

        Args:
            index: VectorSearchIndex instance.
            queries: 2D query array of shape (Q, D).
            k: Top-K items to search.
            nprobe: Centroid probe count for IVF indices.
            warmup_runs: Untimed warm-up iterations.
            benchmark_runs: Timed measurement iterations.

        Returns:
            Dictionary with latency statistics (ms) and QPS.
        """
        num_queries = len(queries)
        kwargs = {}
        if nprobe is not None:
            kwargs["nprobe"] = nprobe

        # Warm-up pass
        for _ in range(warmup_runs):
            _ = index.search(queries, k=k, **kwargs)

        # Timed passes
        latencies_per_query_ms = []
        batch_times_sec = []

        for _ in range(benchmark_runs):
            t_start = time.perf_counter()
            _ = index.search(queries, k=k, **kwargs)
            t_elapsed = time.perf_counter() - t_start

            batch_times_sec.append(t_elapsed)
            latencies_per_query_ms.append((t_elapsed / num_queries) * 1000.0)

        mean_batch_time = float(np.mean(batch_times_sec))
        qps = float(num_queries / mean_batch_time) if mean_batch_time > 0 else 0.0

        mean_latency = float(np.mean(latencies_per_query_ms))
        std_latency = float(np.std(latencies_per_query_ms))
        p50_latency = float(np.percentile(latencies_per_query_ms, 50))
        p95_latency = float(np.percentile(latencies_per_query_ms, 95))
        p99_latency = float(np.percentile(latencies_per_query_ms, 99))

        return {
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "qps": qps,
        }

    @classmethod
    def evaluate_ann_retrieval(
        cls,
        ann_index: VectorSearchIndex,
        exact_index: ExactSearchIndex,
        queries: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
        benchmark_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Evaluate a single ANN index configuration against the exact ground truth.

        Args:
            ann_index: Approximate index (e.g. IVFIndex).
            exact_index: ExactSearchIndex instance.
            queries: 2D array of query vectors.
            k: Top-K items to retrieve.
            nprobe: Probe setting for IVFIndex.
            benchmark_runs: Timed iterations for latency measurement.

        Returns:
            Dictionary containing Recall Retention, Top-1 Accuracy, Latency, QPS, and Speedup.
        """
        # Ground truth search
        exact_ids, _ = exact_index.search(queries, k=k)
        exact_latency_stats = cls.benchmark_query_latency(
            exact_index, queries, k=k, benchmark_runs=benchmark_runs
        )

        # ANN search
        kwargs = {}
        if nprobe is not None:
            kwargs["nprobe"] = nprobe

        ann_ids, _ = ann_index.search(queries, k=k, **kwargs)
        ann_latency_stats = cls.benchmark_query_latency(
            ann_index, queries, k=k, nprobe=nprobe, benchmark_runs=benchmark_runs
        )

        recall_retention = cls.recall_retention_at_k(exact_ids, ann_ids, k=k)
        top1_acc = cls.top_1_accuracy(exact_ids, ann_ids)

        speedup = (
            exact_latency_stats["mean_latency_ms"] / ann_latency_stats["mean_latency_ms"]
            if ann_latency_stats["mean_latency_ms"] > 0
            else 1.0
        )

        return {
            f"recall_retention@{k}": recall_retention,
            "top1_accuracy": top1_acc,
            "ann_mean_latency_ms": ann_latency_stats["mean_latency_ms"],
            "ann_p95_latency_ms": ann_latency_stats["p95_latency_ms"],
            "ann_qps": ann_latency_stats["qps"],
            "exact_mean_latency_ms": exact_latency_stats["mean_latency_ms"],
            "speedup_factor": speedup,
        }

    @classmethod
    def benchmark_tradeoff_sweep(
        cls,
        ann_index: IVFIndex,
        exact_index: ExactSearchIndex,
        queries: np.ndarray,
        k: int = 10,
        nprobe_list: Optional[Sequence[int]] = None,
        benchmark_runs: int = 3,
    ) -> pd.DataFrame:
        """
        Perform a hyperparameter sweep over nprobe values to trace the Pareto frontier
        of Recall Retention vs. Latency / Throughput.

        Args:
            ann_index: IVFIndex instance.
            exact_index: ExactSearchIndex instance.
            queries: 2D array of query vectors.
            k: Top-K cutoff.
            nprobe_list: List of nprobe integers to test (defaults to powers of 2 up to nlist).
            benchmark_runs: Number of benchmark runs per configuration.

        Returns:
            pd.DataFrame summarizing the Pareto trade-off table.
        """
        if nprobe_list is None:
            probes = [1, 2, 4, 8, 16, 32]
            nprobe_list = [p for p in probes if p <= ann_index.nlist]
            if ann_index.nlist not in nprobe_list:
                nprobe_list.append(ann_index.nlist)

        # Baseline ground truth
        exact_ids, _ = exact_index.search(queries, k=k)
        exact_lat = cls.benchmark_query_latency(exact_index, queries, k=k, benchmark_runs=benchmark_runs)
        exact_ms = exact_lat["mean_latency_ms"]

        rows = []
        for p in nprobe_list:
            ann_ids, _ = ann_index.search(queries, k=k, nprobe=p)
            recall_ret = cls.recall_retention_at_k(exact_ids, ann_ids, k=k)
            top1_acc = cls.top_1_accuracy(exact_ids, ann_ids)
            lat_stats = cls.benchmark_query_latency(
                ann_index, queries, k=k, nprobe=p, benchmark_runs=benchmark_runs
            )

            ann_ms = lat_stats["mean_latency_ms"]
            speedup = exact_ms / ann_ms if ann_ms > 0 else 1.0

            rows.append({
                "nprobe": p,
                "nlist": ann_index.nlist,
                f"recall_retention@{k}": recall_ret,
                "top1_accuracy": top1_acc,
                "latency_ms": ann_ms,
                "p95_latency_ms": lat_stats["p95_latency_ms"],
                "qps": lat_stats["qps"],
                "speedup_vs_exact": speedup,
            })

        return pd.DataFrame(rows)


# =====================================================================
# Standalone Demonstration & Verification
# =====================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 80)
    print("DAY 4: APPROXIMATE NEAREST NEIGHBOR (ANN) BENCHMARKING HARNESS")
    print("=" * 80)

    # 1. Synthesize candidate items and query embeddings
    num_items = 3000
    num_queries = 150
    dim = 64
    k = 10
    nlist = 32

    print(f"\n[1] Generating synthetic embedding space:")
    print(f"    - Candidate Items Catalog : {num_items} items")
    print(f"    - Query Embedding Batch   : {num_queries} queries")
    print(f"    - Embedding Dimension     : {dim}D (unit sphere / cosine)")
    print(f"    - Retrieval Cutoff (K)    : Top-{k}")

    np.random.seed(42)
    item_ids = [f"item_{i:04d}" for i in range(num_items)]
    item_embeddings = np.random.randn(num_items, dim).astype(np.float32)
    query_embeddings = np.random.randn(num_queries, dim).astype(np.float32)

    # 2. Build Exact Ground-Truth Baseline Index
    print("\n[2] Building ExactSearchIndex (Exhaustive Brute-Force Baseline)...")
    exact_index = ExactSearchIndex(normalize=True)
    exact_index.fit(item_ids, item_embeddings)
    exact_perf = ANNEvaluator.benchmark_query_latency(exact_index, query_embeddings, k=k)
    print(f"    - Exact Search Mean Latency : {exact_perf['mean_latency_ms']:.3f} ms / query")
    print(f"    - Exact Search Throughput   : {exact_perf['qps']:.1f} QPS")

    # 3. Build IVF Approximate Nearest Neighbor Index
    print(f"\n[3] Building IVFIndex (nlist={nlist} Voronoi clusters)...")
    t0 = time.perf_counter()
    ivf_index = IVFIndex(nlist=nlist, nprobe=1, normalize=True, random_state=42)
    ivf_index.fit(item_ids, item_embeddings)
    build_time = time.perf_counter() - t0
    print(f"    - IVF Index Build & Clustering Time: {build_time:.4f} seconds")

    # 4. Sweep nprobe Pareto Frontier Trade-Off
    print(f"\n[4] Sweeping nprobe Pareto Frontier Trade-Off (Recall Retention vs Latency):")
    probes_to_test = [1, 2, 4, 8, 16, 32]
    tradeoff_df = ANNEvaluator.benchmark_tradeoff_sweep(
        ann_index=ivf_index,
        exact_index=exact_index,
        queries=query_embeddings,
        k=k,
        nprobe_list=probes_to_test,
        benchmark_runs=3,
    )

    print("\n" + tradeoff_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANN Retrieval & Latency/Recall Trade-off Benchmark Completed Successfully!")
    print("=" * 80 + "\n")
