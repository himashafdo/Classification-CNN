"""
src/time_epoch.py
Measures training time per epoch for Model A (model_a_7) and Model B (model_b_3)
on the SAME machine, with the same data, batch size, optimizer (Adam 1e-3) and
class weights as comparison.py, so the "time per epoch" row of the A vs B table
is a fair comparison. Epoch 1 is excluded (graph tracing + dataset caching).
"""

import time
import platform
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight

from model_a_builds.model_a_7 import model_a
from model_b_builds.model_b_3 import model_b

# ---------------- Config ----------------
DATA_DIR    = Path("data/RealWaste_split")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
IMG_SIZE, BATCH_SIZE = (64, 64), 64
EPOCHS, SEED = 6, 42           # 1 warm-up epoch + 5 timed epochs
AUTOTUNE = tf.data.AUTOTUNE

# ---------------- Data (same as comparison.py) ----------------
train_ds = image_dataset_from_directory(DATA_DIR/"train", image_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, label_mode="int", seed=SEED)
val_ds   = image_dataset_from_directory(DATA_DIR/"val", image_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, label_mode="int", shuffle=False)
class_names = train_ds.class_names

train_labels = np.concatenate([[i] * len(list((DATA_DIR/"train"/c).glob("*")))
                               for i, c in enumerate(class_names)])
cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(cw))

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)


# records wall-clock time of every epoch
class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._start)


rows = []
for tag, base in [("A", model_a), ("B", model_b)]:
    print(f"\n===== Timing Model {tag} =====")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.models.clone_model(base)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    timer = EpochTimer()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              class_weight=class_weights, callbacks=[timer], verbose=2)

    rows.append({
        "Model": tag,
        "Trainable params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "Time/epoch (s), mean of epochs 2-6": np.mean(timer.epoch_times[1:]),
        "Std (s)": np.std(timer.epoch_times[1:]),
        "Machine": f"{platform.system()} {platform.machine()} {platform.processor()}",
    })

summary = pd.DataFrame(rows).round(3)
print("\n", summary.to_string(index=False))
summary.to_csv(RESULTS_DIR / "epoch_timing_same_machine.csv", index=False)