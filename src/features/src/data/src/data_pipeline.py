import numpy as np
import tensorflow as tf


def create_data_pipeline(
    features,
    labels,
    batch_size=32,
    train_ratio=0.8,
    buffer_size=1000,
    seed=42,
):
    """Converts raw data into optimized Train and Validation tf.data.Datasets."""
    # 1. Convert prepared samples into a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    total_samples = len(features)
    train_size = int(total_samples * train_ratio)

    # Shuffle once before splitting (reshuffle_each_iteration=False prevents validation leakage)
    dataset = dataset.shuffle(
        buffer_size=total_samples, seed=seed, reshuffle_each_iteration=False
    )

    # 2. Implement train/validation split logic directly in the pipeline
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)

    # 3. Optimize Training Pipeline: cache -> shuffle -> batch -> prefetch
    train_ds = (
        train_ds.cache()
        .shuffle(buffer_size=buffer_size, seed=seed)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # 4. Optimize Validation Pipeline: batch -> cache -> prefetch (no shuffle needed)
    val_ds = (
        val_ds.batch(batch_size)
        .cache()
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


# --- Execution and Verification ---
if __name__ == "__main__":
    print("=" * 50)
    print("Running tf.data Pipeline Verification...")
    print("=" * 50)

    # 1. Create simulated samples (replace with your actual Day 2 output if available)
    num_samples = 1000
    num_features = 8  # e.g., user_id, item_id, context features

    features_sample = np.random.rand(num_samples, num_features).astype(
        np.float32
    )
    labels_sample = np.random.randint(0, 2, size=(num_samples, 1)).astype(
        np.int32
    )

    # 2. Build the datasets
    batch_size = 32
    train_dataset, val_dataset = create_data_pipeline(
        features=features_sample,
        labels=labels_sample,
        batch_size=batch_size,
        train_ratio=0.8,
    )

    # 3. Test and print results
    print("\n[SUCCESS] Pipeline created.")

    # Inspect 1 batch from the Training Set
    for x_batch, y_batch in train_dataset.take(1):
        print(f"\n--- Train Batch Sample ---")
        print(f"Feature Batch Shape : {x_batch.shape}  (Expected: [32, 8])")
        print(f"Label Batch Shape   : {y_batch.shape}  (Expected: [32, 1])")
        print(f"First 2 rows of features:\n{x_batch[:2].numpy()}")
        print(f"First 2 rows of labels:\n{y_batch[:2].numpy()}")

    # Count total batches
    train_batch_count = tf.data.experimental.cardinality(train_dataset).numpy()
    val_batch_count = tf.data.experimental.cardinality(val_dataset).numpy()

    print(f"\n--- Dataset Split Summary ---")
    print(
        f"Total Samples: {num_samples} (800 Train / 200 Val with 80/20 split)"
    )
    print(f"Train batches (batch_size={batch_size}): {train_batch_count}")
    print(f"Val batches   (batch_size={batch_size}): {val_batch_count}")
    print("=" * 50)