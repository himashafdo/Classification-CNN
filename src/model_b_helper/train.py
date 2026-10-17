"""
src/train.py
Final training run for a custom model, using the same settings as Model A in the report:
  Adam (lr=1e-3), batch size 64, 30 epochs, seed 42, class-weighted loss,
  LR halved after 3 epochs without val-loss improvement, best-val-loss weights kept.

Outputs (in results/):
  model_<tag>_30ep_<opt>.weights.h5        best-val-loss weights (used by evaluate.py)
  model_<tag>_30ep_<opt>_history.csv       per-epoch train/val loss and accuracy
  model_<tag>_30ep_<opt>_loss.png          training vs validation loss curve (for the report)
  model_<tag>_30ep_optimizer_comparison.csv summary: best epoch, min val loss, time/epoch, ...

Run from the project root:  python src/<subfolder>/train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # add src/ so model_*_builds can be imported

import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight

from model_a_builds.model_a_7 import model_a
from model_b_builds.model_b_3 import model_b

# ---------------- Config ----------------
MODEL_TAG   = "b"                                   # "a" or "b"
BASE_MODEL  = model_a if MODEL_TAG == "a" else model_b
RUN_NAME    = f"model_{MODEL_TAG}_30ep"
DATA_DIR    = Path("data/RealWaste_split")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
IMG_SIZE, BATCH_SIZE = (64, 64), 64
EPOCHS, SEED = 30, 42
VAL_ACC_TARGET = 0.70

# Adam is the optimizer used in the report. Set True to also run SGD and SGD+Momentum.
RUN_SGD_COMPARISON = False

AUTOTUNE = tf.data.AUTOTUNE

# ---------------- Parameter budget check ----------------
trainable = int(np.sum([np.prod(v.shape) for v in BASE_MODEL.trainable_weights]))
print(f"Model {MODEL_TAG.upper()} trainable params: {trainable:,}")
if MODEL_TAG == "b" and trainable > 100_000:
    raise SystemExit("Model B exceeds the 100,000 trainable-parameter limit.")

# ---------------- Data ----------------
train_ds = image_dataset_from_directory(DATA_DIR / "train", image_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, label_mode="int", seed=SEED)
val_ds   = image_dataset_from_directory(DATA_DIR / "val", image_size=IMG_SIZE,
                                        batch_size=BATCH_SIZE, label_mode="int", shuffle=False)
class_names = train_ds.class_names

# Class weights from file counts ("balanced": w_c = N / (C * N_c))
train_labels = np.concatenate([[i] * len(list((DATA_DIR / "train" / c).glob("*")))
                               for i, c in enumerate(class_names)])
cw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# ---------------- Experiments ----------------
experiments = {"Adam (lr=1e-3)": lambda: tf.keras.optimizers.Adam(1e-3)}
if RUN_SGD_COMPARISON:
    experiments["SGD (lr=1e-2)"]         = lambda: tf.keras.optimizers.SGD(1e-2, momentum=0.0)
    experiments["SGD+Mom 0.9 (lr=1e-2)"] = lambda: tf.keras.optimizers.SGD(1e-2, momentum=0.9)

histories, rows = {}, []

for name, make_opt in experiments.items():
    print(f"\n===== Model {MODEL_TAG.upper()}: {name} =====")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)                 # identical init across runs
    model = tf.keras.models.clone_model(BASE_MODEL)      # fresh weights, same architecture
    model.compile(optimizer=make_opt(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    tag = name.split(" (")[0].replace("+", "_").replace(" ", "_").lower()   # adam / sgd / sgd_mom_0.9
    best_path = str(RESULTS_DIR / f"{RUN_NAME}_{tag}.weights.h5")
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss",
                                           save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]

    t0 = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                  class_weight=class_weights, callbacks=cbs, verbose=2)
    train_time = time.time() - t0
    hist = h.history
    histories[name] = hist

    # Save per-epoch history (so curves can be re-plotted without retraining)
    pd.DataFrame(hist).rename_axis("epoch").to_csv(RESULTS_DIR / f"{RUN_NAME}_{tag}_history.csv")

    # Training vs validation loss for this run (figure for the report)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, EPOCHS + 1), hist["loss"], label="Training")
    plt.plot(range(1, EPOCHS + 1), hist["val_loss"], label="Validation")
    plt.title(f"Model {MODEL_TAG.upper()} ({name}): loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{RUN_NAME}_{tag}_loss.png", dpi=150)
    plt.close()

    best_ep = int(np.argmin(hist["val_loss"])) + 1
    reached = next((i + 1 for i, a in enumerate(hist["val_accuracy"]) if a >= VAL_ACC_TARGET), None)

    rows.append({
        "Optimizer": name,
        "Best val-loss epoch": best_ep,
        "Min val loss": min(hist["val_loss"]),
        "Final train loss": hist["loss"][-1],
        "Final train acc": hist["accuracy"][-1],
        "Best val acc": max(hist["val_accuracy"]),
        f"Epochs to val acc >= {VAL_ACC_TARGET:.0%}": reached if reached else "not reached",
        "Time/epoch (s)": train_time / EPOCHS,
    })

summary = pd.DataFrame(rows).round(4)
print("\n", summary.to_string(index=False))
summary.to_csv(RESULTS_DIR / f"{RUN_NAME}_optimizer_comparison.csv", index=False)

# Combined optimizer curves (only useful when more than one optimizer was run)
if len(histories) > 1:
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for name, hist in histories.items():
        ax[0].plot(hist["loss"], label=name)
        ax[1].plot(hist["val_loss"], label=name)
        ax[2].plot(hist["val_accuracy"], label=name)
    for a, t in zip(ax, ["Training loss", "Validation loss", "Validation accuracy"]):
        a.set_title(f"Model {MODEL_TAG.upper()}: {t}"); a.set_xlabel("Epoch"); a.grid(alpha=0.3)
    ax[0].set_ylabel("Loss"); ax[2].set_ylabel("Accuracy"); ax[2].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{RUN_NAME}_optimizer_curves.png", dpi=150)
    plt.close()

print(f"\nDone. Outputs saved in {RESULTS_DIR}/")