"""
Unit tests for Day 5: Context-Slice & Recommendation Diversity/Coverage Evaluation.
Tests beyond-accuracy metrics (Catalog Coverage, Intra-List Diversity, Prediction Entropy,
Gini Coefficient, Novelty) and SliceEvaluator context partitioning and disparity detection.
"""

import pytest
import math
import numpy as np
import pandas as pd

from src.evaluation import (
    catalog_coverage,
    intra_list_diversity,
    prediction_entropy,
    gini_coefficient,
    novelty_at_k,
    SliceEvaluator,
)


# =====================================================================
# 1. Tests for Beyond-Accuracy Metrics
# =====================================================================

def test_catalog_coverage():
    """Test catalog coverage calculation and cutoff sensitivity."""
    catalog = ["item_1", "item_2", "item_3", "item_4", "item_5"]

    # 1. Zero coverage when predictions are empty or k=0
    assert catalog_coverage([], catalog, k=5) == 0.0
    assert catalog_coverage([["item_1"]], catalog, k=0) == 0.0

    # 2. Partial coverage (3 distinct items recommended in top-2 across 2 sessions)
    preds = [
        ["item_1", "item_2", "item_5"],  # top-2: item_1, item_2
        ["item_2", "item_3", "item_4"],  # top-2: item_2, item_3
    ]
    # In top-2, unique items are {item_1, item_2, item_3} -> 3 / 5 = 0.60
    assert np.isclose(catalog_coverage(preds, catalog, k=2), 0.60)

    # In top-3, unique items are {item_1, item_2, item_3, item_4, item_5} -> 5 / 5 = 1.0
    assert np.isclose(catalog_coverage(preds, catalog, k=3), 1.0)


def test_intra_list_diversity_embeddings():
    """Test Intra-List Diversity using vector embeddings and cosine distance."""
    # Orthogonal embeddings: dot product is 0.0, cosine similarity is 0.0
    # Normalized cosine distance = (1 - 0) / 2 = 0.5
    embs = {
        "item_a": np.array([1.0, 0.0]),
        "item_b": np.array([0.0, 1.0]),
        "item_c": np.array([1.0, 0.0]),  # Identical to item_a
    }

    # Session 1: Orthogonal items -> ILD should be 0.5
    preds_ortho = [["item_a", "item_b"]]
    assert np.isclose(intra_list_diversity(preds_ortho, item_embeddings=embs, k=2), 0.5)

    # Session 2: Identical items -> ILD should be 0.0
    preds_identical = [["item_a", "item_c"]]
    assert np.isclose(intra_list_diversity(preds_identical, item_embeddings=embs, k=2), 0.0)

    # Session with opposite items: dot product is -1.0, distance is 1.0
    embs_opposite = {
        "pos": np.array([1.0, 0.0]),
        "neg": np.array([-1.0, 0.0]),
    }
    assert np.isclose(intra_list_diversity([["pos", "neg"]], item_embeddings=embs_opposite, k=2), 1.0)


def test_intra_list_diversity_categories():
    """Test Intra-List Diversity using categorical genre/department tags."""
    categories = {
        "item_1": "Shoes",
        "item_2": "Shirts",
        "item_3": "Shoes",
    }

    # Diverse list: Shoes and Shirts (mismatch distance = 1.0)
    assert np.isclose(intra_list_diversity([["item_1", "item_2"]], item_categories=categories, k=2), 1.0)

    # Homogeneous list: Shoes and Shoes (mismatch distance = 0.0)
    assert np.isclose(intra_list_diversity([["item_1", "item_3"]], item_categories=categories, k=2), 0.0)


def test_prediction_entropy():
    """Test Shannon entropy calculation of recommendation frequency distribution."""
    # 1. Single item repeated across all sessions -> Entropy = 0.0
    preds_monopoly = [["item_1"], ["item_1"], ["item_1"], ["item_1"]]
    assert np.isclose(prediction_entropy(preds_monopoly, k=1), 0.0)

    # 2. Four distinct items recommended with equal frequency -> Entropy = log2(4) = 2.0
    preds_uniform = [["item_1"], ["item_2"], ["item_3"], ["item_4"]]
    assert np.isclose(prediction_entropy(preds_uniform, k=1), 2.0)


def test_gini_coefficient():
    """Test Gini coefficient inequality index."""
    catalog = ["item_1", "item_2", "item_3", "item_4"]

    # 1. Perfectly equal distribution across catalog items -> Gini should be 0.0
    preds_equal = [["item_1", "item_2"], ["item_3", "item_4"]]
    assert np.isclose(gini_coefficient(preds_equal, catalog_items=catalog, k=2), 0.0)

    # 2. Highly concentrated: only 1 item recommended repeatedly, rest 0 frequency
    preds_concentrated = [["item_1"]] * 100
    gini_val = gini_coefficient(preds_concentrated, catalog_items=catalog, k=1)
    # With 4 items where 1 has 100 and 3 have 0, Gini is 0.75
    assert np.isclose(gini_val, 0.75)


def test_novelty_at_k():
    """Test Novelty (Self-Information) metric."""
    pops = {
        "popular_item": 0.50,   # -log2(0.50) = 1.0 bit
        "niche_item": 0.03125,  # -log2(1/32) = 5.0 bits
    }

    # Popular item recommendation has lower novelty
    assert np.isclose(novelty_at_k([["popular_item"]], item_popularities=pops, k=1), 1.0)

    # Niche item recommendation has higher novelty
    assert np.isclose(novelty_at_k([["niche_item"]], item_popularities=pops, k=1), 5.0)


# =====================================================================
# 2. Tests for SliceEvaluator Pipeline
# =====================================================================

@pytest.fixture
def sample_slice_data():
    """Fixture providing mock evaluation sessions across season and weekend context."""
    catalog = ["item_1", "item_2", "item_3", "item_4", "item_5"]
    sessions = [
        # Winter sessions (high performance)
        {"session_id": "s1", "actual": ["item_1"], "predicted": ["item_1", "item_2"], "context": {"season": "Winter", "is_weekend": 1}},
        {"session_id": "s2", "actual": ["item_2"], "predicted": ["item_2", "item_3"], "context": {"season": "Winter", "is_weekend": 0}},
        # Summer sessions (low performance)
        {"session_id": "s3", "actual": ["item_3"], "predicted": ["item_1", "item_5"], "context": {"season": "Summer", "is_weekend": 1}},
        {"session_id": "s4", "actual": ["item_4"], "predicted": ["item_1", "item_2"], "context": {"season": "Summer", "is_weekend": 0}},
    ]
    return {"catalog": catalog, "sessions": sessions}


def test_slice_evaluator_grouping(sample_slice_data):
    """Test SliceEvaluator groups sessions and calculates correct per-slice metrics."""
    catalog = sample_slice_data["catalog"]
    evaluator = SliceEvaluator(catalog_items=catalog)

    for s in sample_slice_data["sessions"]:
        evaluator.add_session(s["session_id"], s["actual"], s["predicted"], s["context"])

    assert evaluator.num_sessions == 4

    # Evaluate season slice
    df_season = evaluator.evaluate_slice("season", k_values=[1, 2])
    assert len(df_season) == 2
    assert set(df_season["slice_value"]) == {"Winter", "Summer"}

    winter_row = df_season[df_season["slice_value"] == "Winter"].iloc[0]
    summer_row = df_season[df_season["slice_value"] == "Summer"].iloc[0]

    # Winter has 100% recall@1 (hits on rank 1)
    assert winter_row["recall@1"] == 1.0
    assert winter_row["ndcg@1"] == 1.0

    # Summer has 0% recall@1 and recall@2
    assert summer_row["recall@1"] == 0.0
    assert summer_row["recall@2"] == 0.0


def test_slice_evaluator_disparity(sample_slice_data):
    """Test SliceEvaluator detects and flags severe contextual performance disparity."""
    evaluator = SliceEvaluator(catalog_items=sample_slice_data["catalog"])
    for s in sample_slice_data["sessions"]:
        evaluator.add_session(s["session_id"], s["actual"], s["predicted"], s["context"])

    df_season = evaluator.evaluate_slice("season", k_values=[1])
    disparity = evaluator.compute_slice_disparity(df_season, metric="ndcg@1", disparity_threshold=0.60)

    assert disparity["metric"] == "ndcg@1"
    assert disparity["max_slice"] == "Winter"
    assert disparity["min_slice"] == "Summer"
    assert disparity["max_value"] == 1.0
    assert disparity["min_value"] == 0.0
    assert disparity["disparity_ratio"] == 0.0
    # Ratio 0.0 is below threshold 0.60 -> flagged True
    assert disparity["disparity_flagged"] is True


def test_slice_evaluator_load_from_dataframe():
    """Test loading sessions directly from a pandas DataFrame."""
    df = pd.DataFrame({
        "cust_id": ["u1", "u2", "u3"],
        "ground_truth": ["item_1", "item_2", "item_3"],
        "recommendations": [["item_1", "item_2"], ["item_3", "item_4"], ["item_3", "item_5"]],
        "season": ["Winter", "Spring", "Spring"],
        "channel": [1, 2, 2],
    })

    evaluator = SliceEvaluator()
    evaluator.load_from_dataframe(
        df=df,
        actual_col="ground_truth",
        predicted_col="recommendations",
        context_cols=["season", "channel"],
        session_id_col="cust_id",
    )

    assert evaluator.num_sessions == 3
    channel_df = evaluator.evaluate_slice("channel", k_values=[1])
    assert len(channel_df) == 2


def test_compare_models_on_slice():
    """Test comparing multiple models broken down across context slices."""
    actuals = [["item_1"], ["item_2"]]
    contexts = [{"season": "Winter"}, {"season": "Summer"}]

    models_predictions = {
        "Neural_Model": [["item_1", "item_2"], ["item_2", "item_1"]],  # 100% hits
        "Random_Baseline": [["item_99", "item_88"], ["item_99", "item_88"]],  # 0% hits
    }

    comp_df = SliceEvaluator.compare_models_on_slice(
        models_predictions=models_predictions,
        actuals=actuals,
        contexts=contexts,
        slice_feature="season",
        k=1,
    )

    assert isinstance(comp_df, pd.DataFrame)
    assert "model" in comp_df.columns
    assert set(comp_df["model"]) == {"Neural_Model", "Random_Baseline"}
    neural_ndcg = comp_df[comp_df["model"] == "Neural_Model"]["ndcg@1"].mean()
    random_ndcg = comp_df[comp_df["model"] == "Random_Baseline"]["ndcg@1"].mean()
    assert neural_ndcg == 1.0
    assert random_ndcg == 0.0
