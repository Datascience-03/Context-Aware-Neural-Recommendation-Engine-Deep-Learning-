"""
Unit tests for Day 2 EvaluationDataset and RetrievalEvaluator.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluation.evaluator import EvaluationDataset, RetrievalEvaluator


@pytest.fixture
def sample_interactions_df():
    return pd.DataFrame({
        "customer_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u4"],
        "article_id": ["a1", "a2", "a3", "a2", "a4", "a1", "a5", "a6"],
        "t_dat": [
            "2026-08-01", "2026-08-02", "2026-08-03",
            "2026-08-01", "2026-08-04",
            "2026-08-02", "2026-08-05",
            "2026-08-03"
        ]
    })


def test_evaluation_dataset_ground_truth(sample_interactions_df):
    eval_ds = EvaluationDataset(sample_interactions_df)

    assert set(eval_ds.get_test_users()) == {"u1", "u2", "u3", "u4"}
    assert eval_ds.get_ground_truth("u1") == ["a1", "a2", "a3"]
    assert eval_ds.get_ground_truth("u2") == ["a2", "a4"]
    assert eval_ds.get_ground_truth("u999") == []
    assert len(eval_ds.all_items) == 6


def test_temporal_split(sample_interactions_df):
    train_ds, test_ds = EvaluationDataset.create_temporal_split(
        sample_interactions_df, time_col="t_dat", test_ratio=0.25
    )

    # Total rows = 8, test ratio = 0.25 -> train = 6, test = 2
    assert len(train_ds.df) == 6
    assert len(test_ds.df) == 2

    # Verify no temporal inversion: max train date <= min test date
    assert train_ds.df["t_dat"].max() <= test_ds.df["t_dat"].min()


def test_leave_k_out_split(sample_interactions_df):
    train_ds, test_ds = EvaluationDataset.create_leave_k_out_split(
        sample_interactions_df, k=1, user_col="customer_id", item_col="article_id", time_col="t_dat"
    )

    # Each user has at least 1 interaction, so test_ds has 1 interaction per user
    test_user_counts = test_ds.df["customer_id"].value_counts()
    for count in test_user_counts:
        assert count == 1

    # u1 latest interaction on 2026-08-03 is a3
    assert test_ds.get_ground_truth("u1") == ["a3"]
    # u1 earlier interactions a1, a2 must be in train
    assert set(train_ds.get_ground_truth("u1")) == {"a1", "a2"}


def test_generate_candidate_pools(sample_interactions_df):
    eval_ds = EvaluationDataset(sample_interactions_df)
    # Total 6 items. u1 has 3 positives (a1, a2, a3). Negatives available = 3 (a4, a5, a6).
    pools = eval_ds.generate_candidate_pools(num_negatives=2, seed=42)

    assert "u1" in pools
    u1_pool = pools["u1"]
    # Ground truth (3) + sampled negatives (2) = 5
    assert len(u1_pool) == 5
    # All true positives must be present in the candidate pool
    for pos_item in ["a1", "a2", "a3"]:
        assert pos_item in u1_pool


def test_retrieval_evaluator_evaluate_predictions():
    ground_truth = {
        "u1": ["a1", "a2"],
        "u2": ["a3"],
    }
    predictions = {
        "u1": ["a1", "a9", "a2"],  # hit at 1, hit at 3
        "u2": ["a9", "a8", "a3"],  # hit at 3
    }

    evaluator = RetrievalEvaluator(k_values=[1, 3])
    metrics = evaluator.evaluate_predictions(ground_truth, predictions)

    assert "recall@1" in metrics
    assert "recall@3" in metrics
    assert "ndcg@3" in metrics
    assert "hit_rate@1" in metrics
    assert "mrr@3" in metrics

    # u1 recall@1 = 1/2 = 0.5; u2 recall@1 = 0/1 = 0 -> mean = 0.25
    assert metrics["recall@1"] == pytest.approx(0.25)
    # u1 recall@3 = 2/2 = 1.0; u2 recall@3 = 1/1 = 1.0 -> mean = 1.0
    assert metrics["recall@3"] == pytest.approx(1.0)
    # u1 hit@1 = 1; u2 hit@1 = 0 -> mean = 0.5
    assert metrics["hit_rate@1"] == pytest.approx(0.5)


def test_retrieval_evaluator_evaluate_embeddings():
    # 2 users, 4 items, embedding dim = 2
    user_ids = ["u1", "u2"]
    item_ids = ["a1", "a2", "a3", "a4"]

    # u1 vector points along x-axis -> will score a1 highest
    # u2 vector points along y-axis -> will score a2 highest
    user_emb = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    item_emb = np.array([
        [1.0, 0.0],    # a1
        [0.0, 1.0],    # a2
        [-1.0, 0.0],   # a3
        [0.0, -1.0],   # a4
    ])

    ground_truth = {
        "u1": ["a1"],
        "u2": ["a2"],
    }

    evaluator = RetrievalEvaluator(k_values=[1, 2])
    metrics = evaluator.evaluate_embeddings(
        user_embeddings=user_emb,
        item_embeddings=item_emb,
        user_ids=user_ids,
        item_ids=item_ids,
        ground_truth=ground_truth,
    )

    # Perfect top-1 retrieval
    assert metrics["recall@1"] == 1.0
    assert metrics["ndcg@1"] == 1.0
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["mrr@1"] == 1.0


def test_retrieval_evaluator_evaluate_scorer():
    ground_truth = {"u1": ["a1"]}
    candidate_pools = {"u1": ["a2", "a1", "a3"]}

    # Mock scoring function that scores a1 highest
    def mock_scorer(user_id, items):
        score_map = {"a1": 10.0, "a2": 5.0, "a3": 1.0}
        return [score_map.get(it, 0.0) for it in items]

    evaluator = RetrievalEvaluator(k_values=[1])
    metrics = evaluator.evaluate_scorer(mock_scorer, ground_truth, candidate_pools)

    assert metrics["recall@1"] == 1.0
    assert metrics["hit_rate@1"] == 1.0


def test_generate_evaluation_report_df():
    ground_truth = {"u1": ["a1", "a2"], "u2": ["a3"]}
    predictions = {"u1": ["a1", "a9"], "u2": ["a9", "a8"]}

    evaluator = RetrievalEvaluator(k_values=[2])
    report_df = evaluator.generate_evaluation_report_df(ground_truth, predictions, k_eval=2)

    assert len(report_df) == 2
    assert "user_id" in report_df.columns
    assert "recall@2" in report_df.columns
    assert "ndcg@2" in report_df.columns
    assert "hit@2" in report_df.columns
