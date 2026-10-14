"""
src/train_b.py
Training script for Model B. Same data, batch size, class weights and
callbacks as train.py (Model A) so the A vs B comparison is fair.

Extra things recorded here because the assignment asks for them:
- training / validation loss curves  -> results/model_b_version_3_loss.png
- training time per epoch            -> printed + saved in the history csv
- model size on disk (KB)            -> printed at the end
"""

from pathlib import Path
import time
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_b_builds.model_b_3 import model_b

TRAIN_VAL_DATA_DIR = Path("data/RealWaste_split")
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE
MODEL_PATH = Path("results/model_b_version_3.keras")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

train_ds = image_dataset_from_directory(
    TRAIN_VAL_DATA_DIR / "train",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int"
)
val_ds = image_dataset_from_directory(
    TRAIN_VAL_DATA_DIR / "val",
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int", shuffle=False
)

class_names = train_ds.class_names

# class weights from file counts (fast)
train_labels = np.concatenate([
    [i] * len(list((TRAIN_VAL_DATA_DIR / "train" / c).glob("*")))
    for i, c in enumerate(class_names)
])
cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# model_b is already compiled inside model_b_3.py with from_logits=True
# (the last layer outputs logits), so it is NOT recompiled here.


# records wall-clock time of every epoch for the A vs B comparison table
class EpochTimer(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._start)


timer = EpochTimer()

cbs = [
    callbacks.ModelCheckpoint(str(MODEL_PATH),
                              monitor="val_accuracy", save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    # start_from_epoch=12 -> earliest possible stop is epoch 20 (12 + patience 8),
    # because the assignment requires at least 20 epochs of training
    callbacks.EarlyStopping(patience=8, restore_best_weights=True, start_from_epoch=12),
    timer,
]

history = model_b.fit(
    train_ds, validation_data=val_ds,
    epochs=50, class_weight=class_weights, callbacks=cbs
)

# save history (loss, accuracy, lr, epoch time) to csv
hist_df = pd.DataFrame(history.history)
hist_df["epoch_time_s"] = timer.epoch_times
hist_df.index.name = "epoch"
hist_df.to_csv(RESULTS_DIR / "model_b_version_3_history.csv")

# first epoch includes graph tracing + dataset caching, so it is excluded
print(f"\nMean time per epoch (excluding epoch 1): "
      f"{np.mean(timer.epoch_times[1:]):.2f} s")

# training / validation loss curves
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model B ver 3 - Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_b_version_3_loss.png")
plt.close()

# model size on disk
# the .keras checkpoint also stores Adam's optimizer state (~3x the weights).
# rebuilding the model from its config gives a copy with no optimizer, so
# saving it gives the deployable size for a fair comparison with Model A
# (do the same for model_a_8)
INFER_PATH = RESULTS_DIR / "model_b_version_3_inference.keras"
infer_model = tf.keras.Sequential.from_config(model_b.get_config())
infer_model.set_weights(model_b.get_weights())
infer_model.save(INFER_PATH)

print(f"Trainable params: {sum(int(np.prod(w.shape)) for w in model_b.trainable_weights):,}")
print(f"Checkpoint size (with optimizer): {MODEL_PATH.stat().st_size / 1024:.1f} KB")
print(f"Inference model size:             {INFER_PATH.stat().st_size / 1024:.1f} KB")
print(f"Raw float32 weights:              {sum(w.numpy().nbytes for w in model_b.weights) / 1024:.1f} KB")