import tensorflow as tf

from src.model.candidate_tower import CandidateTower


def test_candidate_tower_output_shape():
    model = CandidateTower()

    embeddings = {
        "prod_name": tf.random.normal((2, 16)),
        "product_type_name": tf.random.normal((2, 16)),
        "product_group_name": tf.random.normal((2, 16)),
        "graphical_appearance_name": tf.random.normal((2, 16)),
        "colour_group_name": tf.random.normal((2, 16)),
        "department_name": tf.random.normal((2, 16)),
        "index_name": tf.random.normal((2, 16)),
    }

    output = model(embeddings)

    assert output.shape == (2, 32)
def test_candidate_tower_different_batch_size():
    model = CandidateTower()

    embeddings = {
        "prod_name": tf.random.normal((4, 16)),
        "product_type_name": tf.random.normal((4, 16)),
        "product_group_name": tf.random.normal((4, 16)),
        "graphical_appearance_name": tf.random.normal((4, 16)),
        "colour_group_name": tf.random.normal((4, 16)),
        "department_name": tf.random.normal((4, 16)),
        "index_name": tf.random.normal((4, 16)),
    }

    output = model(embeddings)

    assert output.shape == (4, 32)