import tensorflow as tf

from src.model.candidate_tower import CandidateTower


def make_inputs(batch_size=4):
    """Create test inputs matching the CandidateTower interface."""

    return {
        "article_id_idx": tf.constant(
            [1, 2, 3, 4][:batch_size],
            dtype=tf.int32,
        ),
        "product_group_name_idx": tf.constant(
            [1, 2, 3, 4][:batch_size],
            dtype=tf.int32,
        ),
        "colour_group_name_idx": tf.constant(
            [1, 2, 3, 4][:batch_size],
            dtype=tf.int32,
        ),
        "popularity_over_time": tf.constant(
            [10.0, 20.0, 30.0, 40.0][:batch_size],
            dtype=tf.float32,
        ),
    }


def create_model():
    """Create CandidateTower using the project vocabulary sizes."""

    return CandidateTower(
        num_items=19519,
        num_product_groups=14,
        num_colour_groups=50,
        embedding_dim=64,
    )


def test_candidate_tower_output_shape():
    """CandidateTower should produce 64-dimensional embeddings."""

    model = create_model()

    output = model(
        make_inputs(4),
        training=False,
    )

    assert output.shape == (4, 64)


def test_candidate_tower_output_is_l2_normalized():
    """Candidate embeddings should have L2 norm approximately equal to 1."""

    model = create_model()

    output = model(
        make_inputs(4),
        training=False,
    )

    norms = tf.norm(
        output,
        axis=-1,
    )

    assert tf.reduce_all(
        tf.abs(norms - 1.0) < 1e-5
    )


def test_candidate_tower_different_batch_size():
    """CandidateTower should support different batch sizes."""

    model = create_model()

    output = model(
        make_inputs(2),
        training=False,
    )

    assert output.shape == (2, 64)


def test_candidate_tower_no_nan_or_inf():
    """Candidate embeddings should not contain NaN or Inf values."""

    model = create_model()

    output = model(
        make_inputs(4),
        training=False,
    )

    assert not tf.reduce_any(
        tf.math.is_nan(output)
    )

    assert not tf.reduce_any(
        tf.math.is_inf(output)
    )


def test_candidate_tower_embeddings_are_nonzero():
    """Candidate embeddings should not collapse to zero vectors."""

    model = create_model()

    output = model(
        make_inputs(4),
        training=False,
    )

    norms = tf.norm(
        output,
        axis=-1,
    )

    assert tf.reduce_all(
        norms > 0
    )