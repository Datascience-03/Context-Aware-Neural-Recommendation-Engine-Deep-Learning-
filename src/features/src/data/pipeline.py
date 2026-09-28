import numpy as np
import tensorflow as tf


def create_data_pipelines(
    features,
    labels,
    train_ratio=0.8,
    batch_size=32,
    buffer_size=1024,
    seed=42,
):
    """
    Create TensorFlow training and validation datasets.

    Pipeline:
        input -> shuffle -> train/validation split
        -> batch -> cache -> prefetch
    """

    if len(features) != len(labels):
        raise ValueError(
            "features and labels must contain the same number of samples."
        )

    if not 0 < train_ratio < 1:
        raise ValueError(
            "train_ratio must be between 0 and 1."
        )

    if len(features) == 0:
        raise ValueError("Cannot build a pipeline from empty data.")

    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels)
    )

    total_samples = len(features)
    train_size = int(total_samples * train_ratio)

    # Deterministic shuffle before splitting.
    dataset = dataset.shuffle(
        buffer_size=total_samples,
        seed=seed,
        reshuffle_each_iteration=False,
    )

    # Train/validation split.
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    # Training pipeline.
    train_ds = (
        train_ds
        .cache()
        .shuffle(
            buffer_size=min(buffer_size, train_size),
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Validation pipeline.
    val_ds = (
        val_ds
        .cache()
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


if __name__ == "__main__":

    print("=" * 60)
    print("MEMBER 4 - TENSORFLOW TRAINING PIPELINE VERIFICATION")
    print("=" * 60)

    np.random.seed(42)

    num_samples = 1000
    num_features = 8

    features = np.random.rand(
        num_samples,
        num_features
    ).astype(np.float32)

    labels = np.random.randint(
        0,
        2,
        size=(num_samples, 1)
    ).astype(np.float32)

    train_ds, val_ds = create_data_pipelines(
        features,
        labels,
        train_ratio=0.8,
        batch_size=32,
    )

    train_batches = tf.data.experimental.cardinality(
        train_ds
    ).numpy()

    val_batches = tf.data.experimental.cardinality(
        val_ds
    ).numpy()

    print("\n[SUCCESS] TensorFlow pipeline created.")

    print("\nDataset split:")
    print(f"Total samples : {num_samples}")
    print(f"Train samples : {int(num_samples * 0.8)}")
    print(f"Val samples   : {num_samples - int(num_samples * 0.8)}")

    print("\nBatch information:")
    print(f"Train batches : {train_batches}")
    print(f"Val batches   : {val_batches}")

    for x_batch, y_batch in train_ds.take(1):
        print("\nFirst training batch:")
        print(f"Feature shape : {x_batch.shape}")
        print(f"Label shape   : {y_batch.shape}")

    print("\nTENSORFLOW TRAINING PIPELINE VERIFICATION PASSED")