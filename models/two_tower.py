import tensorflow as tf

class TwoTowerModel(tf.keras.Model):
    def __init__(self, query_tower: tf.keras.Model, candidate_tower: tf.keras.Model, temperature: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.query_tower = query_tower
        self.candidate_tower = candidate_tower
        self.temperature = temperature
        
        # Loss and Metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.CategoricalAccuracy(name="acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy_tracker]

    def call(self, inputs, training=False):
        """
        inputs: dict containing query features and candidate features
        """
        query_emb = self.query_tower(inputs["query_inputs"], training=training)
        candidate_emb = self.candidate_tower(inputs["candidate_inputs"], training=training)
        return query_emb, candidate_emb

    def compute_loss(self, query_emb, candidate_emb):
        # L2-normalize embeddings for cosine similarity
        query_emb = tf.math.l2_normalize(query_emb, axis=1)
        candidate_emb = tf.math.l2_normalize(candidate_emb, axis=1)

        # Compute similarity matrix (batch_size, batch_size)
        similarity = tf.matmul(query_emb, candidate_emb, transpose_b=True) / self.temperature

        # In-batch negatives: target is identity matrix (query i corresponds to candidate i)
        batch_size = tf.shape(query_emb)[0]
        labels = tf.eye(batch_size)

        loss = tf.keras.losses.categorical_crossentropy(labels, similarity, from_logits=True)
        return tf.reduce_mean(loss), similarity, labels

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass
            query_emb, candidate_emb = self(data, training=True)
            loss, similarity, labels = self.compute_loss(query_emb, candidate_emb)

        # Compute gradients across ALL trainable weights (both towers)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Apply gradients via optimizer
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update tracking metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, similarity)

        return {m.name: m.result() for m in self.metrics}