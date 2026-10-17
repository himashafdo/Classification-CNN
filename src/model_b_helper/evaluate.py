"""
src/evaluate.py
Evaluates the saved best-val-loss weights from train.py on the test set:
accuracy, macro precision/recall, per-class report, confusion matrix,
parameter count, FP32 model size and MACs.

Run from the project root (after train.py):  python src/<subfolder>/evaluate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # add src/ so model_*_builds can be imported

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

from model_a_builds.model_a_7 import model_a
from model_b_builds.model_b_3 import model_b

# ---------------- Config ----------------
MODEL_TAG   = "b"                                   # "a" or "b"
BASE_MODEL  = model_a if MODEL_TAG == "a" else model_b
RUN_NAME    = f"model_{MODEL_TAG}_30ep"             # must match train.py
WEIGHT_TAGS = ["adam", "sgd", "sgd_mom_0.9"]        # missing ones are skipped
TEST_DIR    = Path("data/RealWaste_split/test")
WEIGHT_DIR  = Path("results")                       # where train.py saved the weights
RESULTS_DIR = Path("results/eval"); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMG_SIZE, BATCH_SIZE = (64, 64), 64


# ---------------- MAC counter ----------------
def count_macs(model):
    """Multiply-accumulates for one 64x64 image (conv, depthwise, separable and dense layers)."""
    total = 0
    for layer in model.layers:
        kind = layer.__class__.__name__
        try:
            inp, out = layer.input.shape, layer.output.shape
        except Exception:
            continue
        if kind == "Conv2D":
            kh, kw = layer.kernel_size
            total += kh * kw * inp[-1] * out[-1] * out[1] * out[2]
        elif kind == "DepthwiseConv2D":
            kh, kw = layer.kernel_size
            total += kh * kw * inp[-1] * layer.depth_multiplier * out[1] * out[2]
        elif kind == "SeparableConv2D":
            kh, kw = layer.kernel_size
            dm = layer.depth_multiplier
            total += (kh * kw * inp[-1] * dm + inp[-1] * dm * out[-1]) * out[1] * out[2]
        elif kind == "Dense":
            total += inp[-1] * out[-1]
    return int(total)


macs = count_macs(BASE_MODEL)
print(f"Model {MODEL_TAG.upper()} MACs: {macs:,} ({macs / 1e6:.2f} M)")

# ---------------- Data ----------------
test_ds = image_dataset_from_directory(TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                       label_mode="int", shuffle=False)
class_names = test_ds.class_names
y_true = np.concatenate([y.numpy() for _, y in test_ds])


def plot_cm(cm, title, path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"), ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ---------------- Evaluate each saved run ----------------
rows = []
for tag in WEIGHT_TAGS:
    weights_path = WEIGHT_DIR / f"{RUN_NAME}_{tag}.weights.h5"
    if not weights_path.exists():
        print(f"Skipping {tag}: {weights_path} not found")
        continue

    print(f"\n===== Model {MODEL_TAG.upper()} | {tag} =====")
    tf.keras.backend.clear_session()
    model = tf.keras.models.clone_model(BASE_MODEL)
    model.load_weights(weights_path)

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(report_txt)
    (RESULTS_DIR / f"{RUN_NAME}_{tag}_classification_report.txt").write_text(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, f"Model {MODEL_TAG.upper()} ({tag}) - Confusion Matrix",
            RESULTS_DIR / f"{RUN_NAME}_{tag}_confusion_matrix.png")

    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    total = model.count_params()

    rows.append({
        "Optimizer": tag,
        "Test accuracy": report["accuracy"],
        "Macro precision": report["macro avg"]["precision"],
        "Macro recall": report["macro avg"]["recall"],
        "Trainable params": trainable,
        "Total params": total,
        "Size FP32 (KB)": total * 4 / 1024,          # real model size (checkpoint also holds optimizer state)
        "MACs (M)": macs / 1e6,
    })

summary = pd.DataFrame(rows).round(4)
print("\n", summary.to_string(index=False))
summary.to_csv(RESULTS_DIR / f"{RUN_NAME}_evaluation_summary.csv", index=False)
print(f"\nDone. Outputs saved in {RESULTS_DIR}/")