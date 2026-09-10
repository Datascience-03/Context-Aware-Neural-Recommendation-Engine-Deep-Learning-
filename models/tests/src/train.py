# train.py
from src.callbacks import get_callbacks

# ... load data and build model ...

callbacks = get_callbacks()

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)
