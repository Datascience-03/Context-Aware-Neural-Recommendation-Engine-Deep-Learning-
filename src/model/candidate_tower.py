import tensorflow as tf


class CandidateTower(tf.keras.Model):
    """Candidate tower for generating item embeddings."""

    def __init__(self, hidden_units=(64, 32), output_dim=32):
        super().__init__()

        self.dense_layers = [
            tf.keras.layers.Dense(units, activation="relu")
            for units in hidden_units
        ]

        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation=None
        )

    def call(self, embeddings):
        """Pass item embeddings through the candidate tower."""

        # Combine all categorical embeddings
        x = tf.concat(
            [embeddings[column] for column in embeddings],
            axis=-1
        )

        # Apply dense layers
        for layer in self.dense_layers:
            x = layer(x)

        # Generate final candidate representation
        return self.output_layer(x)