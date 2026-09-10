"""
Unit tests for Day 3 Baseline Recommendation Models & Benchmark Harness.
"""

import pytest
import pandas as pd
import numpy as np

from src.evaluation.baselines import (
    PopularityRecommender,
    RecentPopularityRecommender,
    UserHistoryRecommender,
    RandomRecommender,
    BenchmarkHarness,
)


@pytest.fixture
def sample_train_df():
    return pd.DataFrame({
        "customer_id": ["u1", "u1", "u2", "u2", "u3", "u1", "u4", "u2"],
        "article_id": ["pop_item", "rec_item", "pop_item", "mid_item", "pop_item", "old_item", "mid_item", "pop_item"],
        "t_dat": [
            "2026-08-01", "2026-08-20", "2026-08-02", "2026-08-10",
            "2026-08-03", "2026-06-01", "2026-08-05", "2026-08-19"
        ]
    })


def test_popularity_recommender(sample_train_df):
    model = PopularityRecommender().fit(sample_train_df)

    # pop_item appears 4 times, mid_item 2 times, rec_item 1, old_item 1
    recs = model.recommend("u_any", k=2)
    assert recs == ["pop_item", "mid_item"]

    # With candidate pool
    candidate_pool = ["old_item", "mid_item", "unknown_item"]
    pool_recs = model.recommend("u_any", k=2, candidate_pool=candidate_pool)
    assert pool_recs == ["mid_item", "old_item"]


def test_recent_popularity_recommender():
    # Item A has 5 purchases 60 days ago
    # Item B has 2 purchases yesterday
    dates_a = ["2026-06-01"] * 5
    dates_b = ["2026-08-20"] * 2

    df = pd.DataFrame({
        "customer_id": [f"u{i}" for i in range(7)],
        "article_id": ["item_old"] * 5 + ["item_recent"] * 2,
        "t_dat": dates_a + dates_b,
    })

    # With half-life of 7 days, item_recent should rank higher than item_old
    rec_model = RecentPopularityRecommender(half_life_days=7.0).fit(df)
    recs = rec_model.recommend("u_test", k=2)
    assert recs[0] == "item_recent"
    assert recs[1] == "item_old"


def test_user_history_recommender(sample_train_df):
    model = UserHistoryRecommender().fit(sample_train_df)

    # u4 only interacted with mid_item in train
    recs = model.recommend("u4", k=2)
    assert recs[0] == "mid_item"
    # Second item should be filled by global popularity (pop_item)
    assert recs[1] == "pop_item"

    # Candidate pool filtering
    candidate_pool = ["rec_item", "old_item", "mid_item"]
    pool_recs = model.recommend("u4", k=2, candidate_pool=candidate_pool)
    assert pool_recs[0] == "mid_item"
    assert len(pool_recs) == 2


def test_random_recommender(sample_train_df):
    model = RandomRecommender(seed=123).fit(sample_train_df)

    pool = ["item_a", "item_b", "item_c", "item_d"]
    recs_1 = model.recommend("u1", k=2, candidate_pool=pool)
    assert len(recs_1) == 2
    assert len(set(recs_1)) == 2
    assert all(r in pool for r in recs_1)

    # Empty pool edge case
    assert model.recommend("u1", k=3, candidate_pool=[]) == []


def test_benchmark_harness(sample_train_df):
    ground_truth = {
        "u1": ["rec_item"],
        "u2": ["pop_item"],
    }
    candidate_pools = {
        "u1": ["rec_item", "pop_item", "old_item"],
        "u2": ["pop_item", "mid_item", "rec_item"],
    }

    pop_model = PopularityRecommender().fit(sample_train_df)
    rand_model = RandomRecommender(seed=42).fit(sample_train_df)

    harness = BenchmarkHarness(ground_truth=ground_truth, candidate_pools=candidate_pools, k_values=[1, 2])
    harness.register_model("Popularity", pop_model)
    harness.register_model("Random", rand_model)

    # Precomputed perfect model
    perfect_preds = {
        "u1": ["rec_item", "old_item"],
        "u2": ["pop_item", "mid_item"],
    }
    harness.register_predictions("PerfectModel", perfect_preds)

    leaderboard = harness.run_benchmark()

    assert len(leaderboard) == 3
    assert "Model" in leaderboard.columns
    assert "recall@1" in leaderboard.columns
    assert "ndcg@1" in leaderboard.columns
    assert "hit_rate@1" in leaderboard.columns

    # PerfectModel should have recall@1 = 1.0
    perfect_row = leaderboard[leaderboard["Model"] == "PerfectModel"].iloc[0]
    assert perfect_row["recall@1"] == 1.0
