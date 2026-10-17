"""
src/evaluate.py
Evaluates saved best-val-loss weights on the test set for each optimizer:
accuracy, precision, recall, confusion matrix, parameter count, size on disk.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

from model_a_builds.model_a_7 import model_a
from model_b_builds.model_b_3 import model_b   # adjust to your actual path

# ---------------- Config ----------------
MODEL_TAG   = "a"                                  # "a" or "b"
BASE_MODEL  = model_a if MODEL_TAG == "a" else model_b
RUN_NAME    = f"model_{MODEL_TAG}_30ep"            # must match the training run
WEIGHT_TAGS = ["adam", "sgd", "sgd_mom_0.9"]       # add sweep tags if needed
TEST_DIR    = Path("data/RealWaste_split/test")
WEIGHT_DIR= Path("results")
RESULTS_DIR = Path("results/new"); RESULTS_DIR.mkdir(exist_ok=True)
IMG_SIZE, BATCH_SIZE = (64, 64), 64

# ---------------- Data ----------------
test_ds = image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="int", shuffle=False)
class_names = test_ds.class_names
print("Classes:", class_names)

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


rows = []
for tag in WEIGHT_TAGS:
    weights_path = WEIGHT_DIR / f"{RUN_NAME}_{tag}.weights.h5"
    if not weights_path.exists():
        print(f"Skipping {tag}: {weights_path} not found")
        continue

    print(f"\n===== Model {MODEL_TAG.upper()} | {tag} =====")
    tf.keras.backend.clear_session()
    model = tf.keras.models.clone_model(BASE_MODEL)    # same architecture, fresh weights
    model.load_weights(weights_path)

    # Predictions
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Metrics
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True)
    print(report_txt)
    (RESULTS_DIR / f"{RUN_NAME}_{tag}_classification_report.txt").write_text(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, f"Model {MODEL_TAG.upper()} ({tag}) - Confusion Matrix",
            RESULTS_DIR / f"{RUN_NAME}_{tag}_confusion_matrix.png")

    # Size: trainable params, float32 estimate, and saved file size
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    total = model.count_params()
    est_kb = total * 4 / 1024
    file_kb = weights_path.stat().st_size / 1024

    rows.append({
        "Optimizer": tag,
        "Test accuracy": report["accuracy"],
        "Macro precision": report["macro avg"]["precision"],
        "Macro recall": report["macro avg"]["recall"],
        "Macro F1": report["macro avg"]["f1-score"],
        "Trainable params": trainable,
        "Total params": total,
        "Est. size float32 (KB)": est_kb,
        "Weights file (KB)": file_kb,
    })

summary = pd.DataFrame(rows).round(4)
print("\n", summary.to_string(index=False))
summary.to_csv(RESULTS_DIR / f"{RUN_NAME}_evaluation_summary.csv", index=False)