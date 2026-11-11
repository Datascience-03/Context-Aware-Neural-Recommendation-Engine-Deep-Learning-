import os
import datetime
import tensorflow as tf

def get_callbacks(checkpoint_dir="checkpoints", log_dir="logs/fit"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=run_log_dir,
            histogram_freq=1,
            update_freq="epoch"
        )
    ]
