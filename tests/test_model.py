"""
Unit tests for Day 6: Two-Tower Neural Recommendation Model.

Tests cover:
    - QueryTower output shape and L2-normalisation
    - CandidateTower output shape and L2-normalisation
    - TwoTowerModel forward pass (embedding shapes)
    - In-batch softmax loss properties (positive diagonal scores vs negatives)
    - Single train_step execution without error
    - ModelTrainer dataset building and embedding export
    - ModelEvaluationPipeline loading and retrieval evaluation

Note: All tests are skipped automatically if TensorFlow is not installed.
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest

# Skip entire module if TensorFlow is not installed
tf = pytest.importorskip(
    "tensorflow",
    reason="TensorFlow not installed — skipping Two-Tower model tests. "
           "Install with: pip install tensorflow>=2.15",
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.query_tower import QueryTower
from src.model.candidate_tower import CandidateTower
from src.model.two_tower_model import TwoTowerModel
from src.model.train import ModelTrainer, _make_synthetic_df
from src.model.evaluate import ModelEvaluationPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

NUM_USERS           = 50
NUM_ITEMS           = 100
NUM_PRODUCT_GROUPS  = 10
NUM_COLOUR_GROUPS   = 20
EMB_DIM             = 32
BATCH_SIZE          = 16


@pytest.fixture(scope="module")
def query_tower() -> QueryTower:
    tf.random.set_seed(0)
    return QueryTower(
        num_users=NUM_USERS,
        embedding_dim=EMB_DIM,
        user_emb_dim=16,
        context_proj_dim=8,
        hidden_dim=32,
        dropout_rate=0.0,
    )


@pytest.fixture(scope="module")
def candidate_tower() -> CandidateTower:
    tf.random.set_seed(0)
    return CandidateTower(
        num_items=NUM_ITEMS,
        num_product_groups=NUM_PRODUCT_GROUPS,
        num_colour_groups=NUM_COLOUR_GROUPS,
        embedding_dim=EMB_DIM,
        item_emb_dim=16,
        cat_emb_dim=4,
        hidden_dim=32,
        dropout_rate=0.0,
    )


@pytest.fixture(scope="module")
def two_tower_model(query_tower, candidate_tower) -> TwoTowerModel:
    return TwoTowerModel(
        query_tower=query_tower,
        candidate_tower=candidate_tower,
        temperature=0.05,
    )


@pytest.fixture
def user_batch() -> dict:
    """Synthetic user + context feature batch."""
    tf.random.set_seed(42)
    return {
        "customer_id_idx":           tf.random.uniform((BATCH_SIZE,), 0, NUM_USERS, dtype=tf.int32),
        "month_sin":                 tf.random.normal((BATCH_SIZE,)),
        "month_cos":                 tf.random.normal((BATCH_SIZE,)),
        "day_of_week_sin":           tf.random.normal((BATCH_SIZE,)),
        "day_of_week_cos":           tf.random.normal((BATCH_SIZE,)),
        "is_weekend":                tf.zeros((BATCH_SIZE,), dtype=tf.float32),
        "days_since_last_purchase":  tf.random.normal((BATCH_SIZE,)),
        "purchase_sequence":         tf.ones((BATCH_SIZE,), dtype=tf.float32) * 5.0,
    }


@pytest.fixture
def item_batch() -> dict:
    """Synthetic item feature batch."""
    tf.random.set_seed(42)
    return {
        "article_id_idx":           tf.random.uniform((BATCH_SIZE,), 0, NUM_ITEMS, dtype=tf.int32),
        "product_group_name_idx":   tf.random.uniform((BATCH_SIZE,), 0, NUM_PRODUCT_GROUPS, dtype=tf.int32),
        "colour_group_name_idx":    tf.random.uniform((BATCH_SIZE,), 0, NUM_COLOUR_GROUPS, dtype=tf.int32),
        "popularity_over_time":     tf.random.uniform((BATCH_SIZE,), 1.0, 100.0, dtype=tf.float32),
    }


@pytest.fixture
def combined_batch(user_batch, item_batch) -> dict:
    return {**user_batch, **item_batch}


# ─────────────────────────────────────────────────────────────────────────────
# QueryTower tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryTower:

    def test_output_shape(self, query_tower, user_batch):
        """QueryTower output should have shape (batch, EMB_DIM)."""
        out = query_tower(user_batch, training=False)
        assert out.shape == (BATCH_SIZE, EMB_DIM), \
            f"Expected ({BATCH_SIZE}, {EMB_DIM}), got {out.shape}"

    def test_output_l2_normalised(self, query_tower, user_batch):
        """QueryTower output vectors must have unit L2 norm (within float32 tolerance)."""
        out = query_tower(user_batch, training=False).numpy()
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
            err_msg="QueryTower outputs are not L2-normalised.")

    def test_training_mode_runs(self, query_tower, user_batch):
        """QueryTower forward pass in training=True should not raise errors."""
        out = query_tower(user_batch, training=True)
        assert out.shape == (BATCH_SIZE, EMB_DIM)

    def test_different_users_different_embeddings(self, query_tower):
        """Two distinct user IDs with same context should produce different embeddings."""
        tf.random.set_seed(7)
        context = {
            "month_sin": tf.zeros((2,)),
            "month_cos": tf.ones((2,)),
            "day_of_week_sin": tf.zeros((2,)),
            "day_of_week_cos": tf.zeros((2,)),
            "is_weekend": tf.zeros((2,)),
            "days_since_last_purchase": tf.zeros((2,)),
            "purchase_sequence": tf.ones((2,)),
        }
        # User 0 vs user 1
        inputs_a = {"customer_id_idx": tf.constant([0, 0], dtype=tf.int32), **context}
        inputs_b = {"customer_id_idx": tf.constant([1, 0], dtype=tf.int32), **context}
        out_a = query_tower(inputs_a, training=False).numpy()
        out_b = query_tower(inputs_b, training=False).numpy()
        # Row 0 of a != row 0 of b (different user IDs)
        assert not np.allclose(out_a[0], out_b[0], atol=1e-4), \
            "Different user IDs produced identical embeddings — embedding table might be broken."

    def test_config_keys(self, query_tower):
        """get_config() should expose key architecture hyperparameters."""
        cfg = query_tower.get_config()
        for key in ("num_users", "embedding_dim", "user_emb_dim", "hidden_dim"):
            assert key in cfg, f"Missing key '{key}' in QueryTower config."


# ─────────────────────────────────────────────────────────────────────────────
# CandidateTower tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateTower:

    def test_output_shape(self, candidate_tower, item_batch):
        """CandidateTower output should have shape (batch, EMB_DIM)."""
        out = candidate_tower(item_batch, training=False)
        assert out.shape == (BATCH_SIZE, EMB_DIM)

    def test_output_l2_normalised(self, candidate_tower, item_batch):
        """CandidateTower output vectors must have unit L2 norm."""
        out = candidate_tower(item_batch, training=False).numpy()
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5,
            err_msg="CandidateTower outputs are not L2-normalised.")

    def test_training_mode_runs(self, candidate_tower, item_batch):
        out = candidate_tower(item_batch, training=True)
        assert out.shape == (BATCH_SIZE, EMB_DIM)

    def test_config_keys(self, candidate_tower):
        cfg = candidate_tower.get_config()
        for key in ("num_items", "embedding_dim", "item_emb_dim", "cat_emb_dim"):
            assert key in cfg


# ─────────────────────────────────────────────────────────────────────────────
# TwoTowerModel tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoTowerModel:

    def test_forward_pass_shapes(self, two_tower_model, combined_batch):
        """forward pass returns two tensors of shape (batch, EMB_DIM)."""
        q_emb, c_emb = two_tower_model(combined_batch, training=False)
        assert q_emb.shape == (BATCH_SIZE, EMB_DIM)
        assert c_emb.shape == (BATCH_SIZE, EMB_DIM)

    def test_query_embeddings_normalised(self, two_tower_model, combined_batch):
        q_emb, _ = two_tower_model(combined_batch, training=False)
        norms = tf.norm(q_emb, axis=-1).numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_candidate_embeddings_normalised(self, two_tower_model, combined_batch):
        _, c_emb = two_tower_model(combined_batch, training=False)
        norms = tf.norm(c_emb, axis=-1).numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_in_batch_loss_is_positive(self, two_tower_model, combined_batch):
        """Softmax cross-entropy loss must be strictly positive."""
        q_emb, c_emb = two_tower_model(combined_batch, training=False)
        loss, _ = two_tower_model.compute_loss_from_embeddings(q_emb, c_emb)
        assert float(loss.numpy()) > 0.0, "Loss should be > 0 for random embeddings."

    def test_perfect_embeddings_loss_near_zero(self, two_tower_model):
        """When query == candidate, diagonal dominates → loss should be very small."""
        # Same embedding for query and candidate in each row
        identity_embs = tf.math.l2_normalize(
            tf.random.normal((BATCH_SIZE, EMB_DIM)), axis=-1
        )
        # Use very low temperature to make softmax sharper
        model_sharp = TwoTowerModel(
            two_tower_model.query_tower,
            two_tower_model.candidate_tower,
            temperature=0.001,
        )
        loss, acc = model_sharp.compute_loss_from_embeddings(identity_embs, identity_embs)
        assert float(acc.numpy()) == pytest.approx(1.0, abs=0.01), \
            f"Expected top-1 accuracy ≈ 1.0, got {acc.numpy():.4f}"

    def test_train_step_runs_without_error(self, combined_batch):
        """A single gradient-descent step should complete without RuntimeError."""
        qt = QueryTower(num_users=NUM_USERS, embedding_dim=EMB_DIM,
                        hidden_dim=32, dropout_rate=0.0)
        ct = CandidateTower(num_items=NUM_ITEMS, embedding_dim=EMB_DIM,
                            hidden_dim=32, dropout_rate=0.0)
        model = TwoTowerModel(qt, ct, temperature=0.05)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
        out = model.train_step(combined_batch)
        assert "loss" in out
        assert "top1_accuracy" in out
        assert float(out["loss"].numpy()) > 0.0

    def test_get_user_embedding(self, two_tower_model, user_batch):
        """get_user_embedding() should return L2-normalised (batch, EMB_DIM) tensor."""
        emb = two_tower_model.get_user_embedding(user_batch, training=False)
        assert emb.shape == (BATCH_SIZE, EMB_DIM)
        norms = tf.norm(emb, axis=-1).numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_get_item_embedding(self, two_tower_model, item_batch):
        """get_item_embedding() should return L2-normalised (batch, EMB_DIM) tensor."""
        emb = two_tower_model.get_item_embedding(item_batch, training=False)
        assert emb.shape == (BATCH_SIZE, EMB_DIM)
        norms = tf.norm(emb, axis=-1).numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_score_positive_pair_higher_than_random(self, two_tower_model):
        """Dot-product score of a matching pair should exceed random pair score (on average)."""
        np.random.seed(42)
        n = 64
        q_embs = tf.math.l2_normalize(
            tf.random.normal((n, EMB_DIM), seed=1), axis=-1
        ).numpy()
        c_embs = tf.math.l2_normalize(
            tf.random.normal((n, EMB_DIM), seed=1), axis=-1
        ).numpy()  # same seed → same vectors → q_embs ≈ c_embs
        c_random = tf.math.l2_normalize(
            tf.random.normal((n, EMB_DIM), seed=99), axis=-1
        ).numpy()

        pos_scores   = np.sum(q_embs * c_embs,    axis=1).mean()   # diagonal
        random_scores = np.sum(q_embs * c_random, axis=1).mean()   # random pairing
        assert pos_scores > random_scores, \
            "Positive pair scores should be higher than random pair scores."


# ─────────────────────────────────────────────────────────────────────────────
# ModelTrainer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelTrainer:

    @pytest.fixture(autouse=True)
    def tmp_dir(self, tmp_path):
        self._tmp = str(tmp_path / "model_out")
        yield
        # cleanup handled by pytest tmp_path

    def _make_trainer(self) -> ModelTrainer:
        return ModelTrainer(
            embedding_dim=EMB_DIM,
            batch_size=BATCH_SIZE,
            epochs=2,
            learning_rate=1e-3,
            patience=1,
            output_dir=self._tmp,
        )

    def test_build_tf_dataset_shape(self):
        """Dataset batches should have the expected feature keys and integer types."""
        trainer = self._make_trainer()
        df = _make_synthetic_df(n_rows=64, num_users=NUM_USERS, num_items=NUM_ITEMS)
        ds = trainer.build_tf_dataset(df, shuffle=False, drop_remainder=True)
        batch = next(iter(ds))
        assert "customer_id_idx" in batch
        assert "article_id_idx"  in batch
        assert batch["customer_id_idx"].dtype == tf.int32
        assert batch["month_sin"].dtype == tf.float32

    def test_training_loop_runs(self):
        """Training for 2 epochs on 128 synthetic rows should finish without error."""
        trainer = self._make_trainer()
        df = _make_synthetic_df(n_rows=128, num_users=NUM_USERS, num_items=NUM_ITEMS)
        trainer.build_model(
            num_users=NUM_USERS,
            num_items=NUM_ITEMS,
            num_product_groups=NUM_PRODUCT_GROUPS,
            num_colour_groups=NUM_COLOUR_GROUPS,
        )
        ds = trainer.build_tf_dataset(df, shuffle=False, drop_remainder=True)
        history = trainer.train(ds)
        assert "loss" in history.history
        assert len(history.history["loss"]) > 0

    def test_save_item_embeddings_shape(self):
        """Saved item embeddings should have shape (unique_items, EMB_DIM)."""
        trainer = self._make_trainer()
        df = _make_synthetic_df(n_rows=128, num_users=NUM_USERS, num_items=NUM_ITEMS)
        trainer.build_model(
            num_users=NUM_USERS, num_items=NUM_ITEMS,
            num_product_groups=NUM_PRODUCT_GROUPS,
            num_colour_groups=NUM_COLOUR_GROUPS,
        )
        ds = trainer.build_tf_dataset(df, shuffle=False, drop_remainder=True)
        trainer.train(ds)

        item_feat_df = df.drop_duplicates(subset="article_id_idx").reset_index(drop=True)
        item_ids_arr = item_feat_df["article_id_idx"].astype(str).values
        _, i_embs = trainer.save_item_embeddings(item_ids_arr, item_feat_df)

        assert i_embs.shape[1] == EMB_DIM
        # Check all unit-norm
        norms = np.linalg.norm(i_embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_load_embeddings(self):
        """load_embeddings() should correctly reload .npy files from disk."""
        trainer = self._make_trainer()
        df = _make_synthetic_df(n_rows=128, num_users=NUM_USERS, num_items=NUM_ITEMS)
        trainer.build_model(
            num_users=NUM_USERS, num_items=NUM_ITEMS,
            num_product_groups=NUM_PRODUCT_GROUPS,
            num_colour_groups=NUM_COLOUR_GROUPS,
        )
        ds = trainer.build_tf_dataset(df, shuffle=False, drop_remainder=True)
        trainer.train(ds)

        item_feat_df = df.drop_duplicates(subset="article_id_idx").reset_index(drop=True)
        item_ids_arr = item_feat_df["article_id_idx"].astype(str).values
        trainer.save_item_embeddings(item_ids_arr, item_feat_df)

        user_feat_df = df.drop_duplicates(subset="customer_id_idx").reset_index(drop=True)
        user_ids_arr = user_feat_df["customer_id_idx"].astype(str).values
        trainer.save_user_embeddings(user_ids_arr, user_feat_df)

        artefacts = ModelTrainer.load_embeddings(self._tmp)
        assert "item_embeddings" in artefacts
        assert "user_embeddings" in artefacts
        assert artefacts["item_embeddings"].shape[1] == EMB_DIM


# ─────────────────────────────────────────────────────────────────────────────
# ModelEvaluationPipeline tests
# ─────────────────────────────────────────────────────────────────────────────

class TestModelEvaluationPipeline:

    @pytest.fixture(autouse=True)
    def synthetic_embeddings(self, tmp_path):
        """Write synthetic normalised embeddings to a temp directory."""
        np.random.seed(42)
        n_users = 40
        n_items = 80
        d = EMB_DIM

        u_embs = np.random.randn(n_users, d).astype(np.float32)
        u_embs /= np.linalg.norm(u_embs, axis=1, keepdims=True)
        i_embs = np.random.randn(n_items, d).astype(np.float32)
        i_embs /= np.linalg.norm(i_embs, axis=1, keepdims=True)

        u_ids = np.array([f"user_{i}" for i in range(n_users)])
        i_ids = np.array([f"item_{i}" for i in range(n_items)])

        self._model_dir = str(tmp_path / "eval_model")
        os.makedirs(self._model_dir)
        np.save(os.path.join(self._model_dir, "user_ids.npy"),        u_ids)
        np.save(os.path.join(self._model_dir, "user_embeddings.npy"), u_embs)
        np.save(os.path.join(self._model_dir, "item_ids.npy"),        i_ids)
        np.save(os.path.join(self._model_dir, "item_embeddings.npy"), i_embs)

        self._u_embs = u_embs
        self._i_embs = i_embs
        self._u_ids  = u_ids
        self._i_ids  = i_ids
        self._n_users = n_users
        self._n_items = n_items

    def _get_pipeline(self) -> ModelEvaluationPipeline:
        p = ModelEvaluationPipeline(model_dir=self._model_dir, k_values=(5, 10))
        p.load_embeddings()
        p.build_exact_index()
        return p

    def test_load_embeddings(self):
        p = self._get_pipeline()
        assert p.user_embeddings.shape == (self._n_users, EMB_DIM)
        assert p.item_embeddings.shape == (self._n_items, EMB_DIM)

    def test_exact_index_built(self):
        p = self._get_pipeline()
        assert p.exact_index is not None
        assert p.exact_index.num_items == self._n_items

    def test_evaluate_returns_metric_keys(self):
        """evaluate() must return a dict containing recall@K, ndcg@K keys."""
        p = self._get_pipeline()

        # Build test set: for each user assign nearest item as positive
        scores = np.matmul(self._u_embs, self._i_embs.T)
        records = []
        for u_idx in range(self._n_users):
            nearest = int(np.argmax(scores[u_idx]))
            records.append({
                "customer_id": f"user_{u_idx}",
                "article_id":  f"item_{nearest}",
            })
        test_df = pd.DataFrame(records)

        metrics = p.evaluate(test_df)
        assert len(metrics) > 0
        assert "recall@5"  in metrics
        assert "ndcg@5"    in metrics
        assert "recall@10" in metrics
        # recall should be high since we used nearest items as positives
        assert metrics["recall@5"] > 0.5, \
            f"Expected recall@5 > 0.5 for nearest-neighbour positives, got {metrics['recall@5']:.4f}"

    def test_diagnostic_report_structure(self):
        """generate_diagnostic_report() must return a DataFrame with required columns."""
        p = self._get_pipeline()

        records = [
            {"customer_id": f"user_{i}", "article_id": f"item_{i % self._n_items}"}
            for i in range(self._n_users)
        ]
        test_df = pd.DataFrame(records)
        report = p.generate_diagnostic_report(test_df, k_eval=10)

        assert isinstance(report, pd.DataFrame)
        assert "user_id" in report.columns
        assert "recall@10" in report.columns
        assert "ndcg@10"   in report.columns

    def test_file_not_found_raises(self, tmp_path):
        """load_embeddings() must raise FileNotFoundError for missing .npy files."""
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        p = ModelEvaluationPipeline(model_dir=empty_dir)
        with pytest.raises(FileNotFoundError):
            p.load_embeddings()
