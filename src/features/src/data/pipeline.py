import tensorflow as tf


def create_data_pipelines(
    features, labels, train_ratio=0.8, batch_size=32, buffer_size=1024, seed=42
):
    """Creates optimized tf.data training and validation pipelines.

    Args:
        features: Input data (NumPy array, tensors, or file paths).
        labels: Target labels (NumPy array or tensors).
        train_ratio: Float, proportion of data for training (default: 0.8).
        batch_size: Int, number of samples per batch.
        buffer_size: Int, buffer size for shuffling.
        seed: Int, random seed for reproducibility.

    Returns:
        train_ds, val_ds: Optimized tf.data.Dataset objects.
    """

    # 1. Convert prepared samples into a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    # Calculate sizes for the train/val split
    total_samples = len(features)
    train_size = int(total_samples * train_ratio)

    # Shuffle the dataset ONCE deterministically before splitting
    # (reshuffle_each_iteration=False prevents validation data leakage)
    dataset = dataset.shuffle(
        buffer_size=total_samples, seed=seed, reshuffle_each_iteration=False
    )

    # 2. Implement train/validation split logic directly within the pipeline
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    # 3. Add performance optimizations to Training set:
    # Order: cache -> shuffle -> batch -> prefetch
    train_ds = (
        train_ds.cache()  # Caches data in memory to prevent I/O bottlenecks
        .shuffle(
            buffer_size=buffer_size, seed=seed
        )  # Shuffles samples per epoch
        .batch(
            batch_size, drop_remainder=False
        )  # Combines consecutive elements into batches
        .prefetch(
            buffer_size=tf.data.AUTOTUNE
        )  # Overlaps data preprocessing & model execution
    )

    # 4. Add performance optimizations to Validation set:
    # Validation does NOT need shuffling
    val_ds = (
        val_ds.batch(batch_size)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


# ==========================================
# Example Usage & Verification:
# ==========================================
if __name__ == "__main__":
    import numpy as np

    # Dummy data representing Day 2 output (e.g., 1000 samples, 10 features, binary labels)
    dummy_features = np.random.randn(1000, 10).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, size=(1000, 1)).astype(np.float32)

    # Build pipelines
    train_dataset, val_dataset = create_data_pipelines(
        dummy_features,
        dummy_labels,
        train_ratio=0.8,
        batch_size=32,
        buffer_size=500,
    )

    # Inspect a single batch
    for x_batch, y_batch in train_dataset.take(1):
        print(f"Features Batch Shape: {x_batch.shape}")  # Output: (32, 10)
        print(f"Labels Batch Shape:   {y_batch.shape}")  # Output: (32, 1)

    print("\nPipeline built successfully!")