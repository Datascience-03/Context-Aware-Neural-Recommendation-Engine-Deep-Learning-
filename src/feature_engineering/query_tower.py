from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import layers


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_CONFIG_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "user_embedding_config.json"
)


USER_CATEGORICAL_FEATURES = [
    "customer_id",
    "club_member_status",
    "fashion_news_frequency",
    "postal_code",
]


USER_NUMERICAL_FEATURES = [
    "FN",
    "Active",
    "days_since_last_purchase",
    "days_since_first_purchase",
    "avg_inter_purchase_days",
    "purchase_sequence",
    "day_of_week",
    "is_weekend",
    "month",
    "quarter",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]


class QueryTower(tf.keras.Model):

    def __init__(
        self,
        embedding_config,
        output_dim=64,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim

        self.embedding_layers = {}

        for feature in USER_CATEGORICAL_FEATURES:

            settings = embedding_config[feature]

            self.embedding_layers[feature] = layers.Embedding(
                input_dim=settings["vocabulary_size"],
                output_dim=settings["embedding_dimension"],
                name=f"{feature}_embedding"
            )

        self.normalization = layers.LayerNormalization(
    axis=-1,
    name="numerical_normalization"
)

        self.dense_1 = layers.Dense(
            128,
            activation="relu",
            name="dense_128"
        )

        self.dropout = layers.Dropout(
            0.2,
            name="dropout"
        )

        self.dense_2 = layers.Dense(
            64,
            activation="relu",
            name="dense_64"
        )

        self.output_layer = layers.Dense(
            output_dim,
            activation=None,
            name="query_embedding"
        )

    def call(
        self,
        inputs,
        training=False
    ):

        embeddings = []

        for feature in USER_CATEGORICAL_FEATURES:

            feature_input = tf.cast(
                inputs[feature],
                tf.int32
            )

            embedded = self.embedding_layers[feature](
                feature_input
            )

            embeddings.append(embedded)

        categorical_vector = tf.concat(
            embeddings,
            axis=-1
        )

        numerical_inputs = tf.cast(
            tf.stack(
                [
                    inputs[feature]
                    for feature in USER_NUMERICAL_FEATURES
                ],
                axis=-1
            ),
            tf.float32
        )

        numerical_vector = self.normalization(
            numerical_inputs
        )

        combined = tf.concat(
            [
                categorical_vector,
                numerical_vector
            ],
            axis=-1
        )

        x = self.dense_1(combined)

        x = self.dropout(
            x,
            training=training
        )

        x = self.dense_2(x)

        query_embedding = self.output_layer(x)

        return query_embedding


def load_embedding_config():

    if not EMBEDDING_CONFIG_FILE.exists():

        raise FileNotFoundError(
            f"Embedding configuration not found: "
            f"{EMBEDDING_CONFIG_FILE}"
        )

    with open(
        EMBEDDING_CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def create_test_inputs(batch_size=4):

    inputs = {}

    for feature in USER_CATEGORICAL_FEATURES:

        inputs[feature] = tf.constant(
            [1] * batch_size,
            dtype=tf.int32
        )

    for feature in USER_NUMERICAL_FEATURES:

        inputs[feature] = tf.zeros(
            [batch_size],
            dtype=tf.float32
        )

    return inputs


def test_query_tower():

    print("=" * 60)
    print("QUERY / USER TOWER TEST")
    print("=" * 60)

    embedding_config = load_embedding_config()

    model = QueryTower(
        embedding_config=embedding_config,
        output_dim=64
    )

    test_inputs = create_test_inputs(
        batch_size=4
    )

    query_embeddings = model(
        test_inputs,
        training=False
    )

    print("\nQuery Tower architecture:")
    model.summary()

    print(
        "\nInput batch size:",
        4
    )

    print(
        "Query embedding shape:",
        query_embeddings.shape
    )

    assert query_embeddings.shape == (
        4,
        64
    )

    assert not tf.reduce_any(
        tf.math.is_nan(query_embeddings)
    )

    print("\nQuery embedding test passed.")

    print(
        "\nQUERY / USER TOWER COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    test_query_tower()