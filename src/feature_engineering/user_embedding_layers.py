from pathlib import Path
import json

import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FILE = (
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


class UserEmbeddingLayers:
    """
    Creates and manages embedding layers for Query/User Tower
    categorical features.
    """

    def __init__(self, config_path=CONFIG_FILE):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Embedding configuration not found: {self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = json.load(file)

        self.layers = self._build_layers()

    def _build_layers(self):
        """
        Build one Keras Embedding layer for each user categorical feature.
        """

        layers = {}

        for feature in USER_CATEGORICAL_FEATURES:

            if feature not in self.config:
                raise KeyError(
                    f"Missing embedding configuration for: {feature}"
                )

            feature_config = self.config[feature]

            vocabulary_size = feature_config["vocabulary_size"]
            embedding_dimension = feature_config[
                "embedding_dimension"
            ]

            layers[feature] = tf.keras.layers.Embedding(
                input_dim=vocabulary_size,
                output_dim=embedding_dimension,
                name=f"{feature}_embedding",
            )

        return layers

    def get_layer(self, feature):
        """
        Return the embedding layer for a specific feature.
        """

        if feature not in self.layers:
            raise KeyError(
                f"Unknown user feature: {feature}"
            )

        return self.layers[feature]

    def embed(self, feature, values):
        """
        Convert integer feature indices into dense embeddings.
        """

        layer = self.get_layer(feature)

        values = tf.convert_to_tensor(
            values,
            dtype=tf.int32
        )

        return layer(values)

    def summary(self):
        """
        Print embedding layer configuration.
        """

        print("\nUser embedding layers:")

        for feature, layer in self.layers.items():

            print(
                f"  {feature}: "
                f"input_dim={layer.input_dim}, "
                f"output_dim={layer.output_dim}"
            )


def test_user_embedding_layers():

    print("=" * 60)
    print("USER EMBEDDING LAYER TEST")
    print("=" * 60)

    embedding_layers = UserEmbeddingLayers()

    embedding_layers.summary()

    # Test one small batch for every feature.
    test_values = {
        "customer_id": [1, 2, 3],
        "club_member_status": [1, 2, 3],
        "fashion_news_frequency": [1, 2, 3],
        "postal_code": [1, 2, 3],
    }

    print("\nTesting embedding outputs:")

    for feature, values in test_values.items():

        output = embedding_layers.embed(
            feature,
            values
        )

        expected_dimension = embedding_layers.config[
            feature
        ]["embedding_dimension"]

        expected_shape = (
            len(values),
            expected_dimension
        )

        actual_shape = tuple(output.shape)

        print(
            f"  {feature}: "
            f"input_shape={len(values)}, "
            f"output_shape={actual_shape}"
        )

        if actual_shape != expected_shape:
            raise ValueError(
                f"Unexpected embedding shape for {feature}: "
                f"{actual_shape}, expected {expected_shape}"
            )

    print("\nAll embedding layers passed the test.")

    print(
        "\nUSER EMBEDDING LAYERS COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    test_user_embedding_layers()