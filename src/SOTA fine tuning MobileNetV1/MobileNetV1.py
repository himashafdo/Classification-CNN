"""

Fine-tunes an ImageNet-pretrained lightweight network (MobileNetV1) on the SAME
64x64 RealWaste train/val/test splits used for Model A and Model B, then evaluates
it on the test set. (The second SOTA model is fine-tuned by a teammate.)

Outputs (in results/sota/):
    <name>_loss_curves.png, <name>_confusion_matrix.png,
    <name>_classification_report.txt, <name>_final.keras, *.tflite,
    sota_results.json   <- numbers you need for the Part 5 table and Part 6 discussion
"""

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to disk, no window needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers


SEED = 42
keras.utils.set_random_seed(SEED)           # reproducible weight init / shuffling

DATA_DIR = Path("data/RealWaste_split")     # same folders the teammate's data_prep.py made
RESULTS_DIR = Path("results/sota")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

ALPHA = 1.0               # MobileNet width multiplier (1.0 = full width)
HEAD_EPOCHS = 10          # phase 1: backbone frozen, train only the new classifier head
FINE_TUNE_EPOCHS = 30     # phase 2: unfreeze (most of) the backbone (early stopping may cut it short)
HEAD_LR = 1e-3            # new randomly-initialised head -> normal Adam LR
FINE_TUNE_LR = 1e-4       # pretrained weights are 10x smaller LR so we don't wreck them
FREEZE_FRACTION = 0.3     # in phase 2 keep the earliest 30% of backbone layers frozen

BACKBONES = {
    "MobileNetV1": keras.applications.MobileNet,
}


# splits identical to A
def load_split(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle,
        seed=SEED,
    )


train_ds = load_split("train", shuffle=True)
val_ds = load_split("val", shuffle=False)
test_ds = load_split("test", shuffle=False)   # shuffle=False so labels line up with predictions
class_names = train_ds.class_names
n_classes = len(class_names)
print("Classes:", class_names)

train_labels = np.concatenate([y.numpy() for _, y in train_ds])
cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=train_labels)
class_weights = dict(enumerate(cw))
print("Class weights:", class_weights)

# Cache the small val/test sets in RAM. We do NOT cache train_ds: caching a shuffled,
# already-batched dataset freezes the batch composition for every epoch.
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)


#building the model
def build_model(name):
    """augmentation -> rescale to [-1,1] -> pretrained backbone -> GAP -> dropout -> logits"""
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )  # same augmentation as Model A; only active during training

    base = BACKBONES[name](
        input_shape=IMG_SIZE + (3,),
        include_top=False,        # drop the 1000-class ImageNet classifier
        weights="imagenet",       # load pretrained weights
        alpha=ALPHA,
    )
    base.trainable = False        # phase 1: frozen

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    # MobileNetV1 was pretrained on inputs scaled to [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)
    x = base(x, training=False)   # training=False keeps BatchNorm in inference mode (see notes)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes)(x)   # raw logits, so loss uses from_logits=True
    model = keras.Model(inputs, outputs, name=name)
    return model, base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


class EpochTimer(callbacks.Callback):
    """Records wall-clock seconds per epoch (needed for the comparison tables)."""

    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.perf_counter() - self._t0)



#Two-phase training

def train_two_phase(name):
    model, base = build_model(name)
    timer = EpochTimer()

    # training only the new head
    print(f"\n=== {name}: phase 1 (frozen backbone) ===")
    compile_model(model, HEAD_LR)
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=[timer, callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2,
    )

    # unfreeze and fine-tune with a small LR
    print(f"\n=== {name}: phase 2 (fine-tuning) ===")
    base.trainable = True
    n_freeze = int(len(base.layers) * FREEZE_FRACTION)
    for layer in base.layers[:n_freeze]:
        layer.trainable = False
    compile_model(model, FINE_TUNE_LR)      # MUST recompile after changing .trainable
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=[
            timer,
            callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
            callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        ],
        verbose=2,
    )

    history = {k: h1.history[k] + h2.history[k]
               for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
    return model, history, len(h1.history["loss"]), timer.times


def plot_curves(history, phase_boundary, name):
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, history["loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].axvline(phase_boundary + 0.5, color="gray", ls="--", label="start fine-tuning")
    ax[0].set(title=f"{name} - loss", xlabel="epoch", ylabel="loss")
    ax[0].legend()
    ax[1].plot(epochs, history["accuracy"], label="train")
    ax[1].plot(epochs, history["val_accuracy"], label="validation")
    ax[1].axvline(phase_boundary + 0.5, color="gray", ls="--")
    ax[1].set(title=f"{name} - accuracy", xlabel="epoch", ylabel="accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_loss_curves.png", dpi=150)
    plt.close(fig)


#evaluation and metrics
def evaluate_on_test(model, name):
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    (RESULTS_DIR / f"{name}_classification_report.txt").write_text(report)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{name} - confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{name}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return acc, prec, rec, f1


def count_params(model):
    total = model.count_params()   # trainable + non-trainable (BatchNorm moving stats)
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return total, trainable


def tflite_size_mb(model, name, quantize):
    """Edge-style size. quantize=True -> dynamic-range int8 weights. Returns None on failure."""
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantize:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        blob = conv.convert()
        path = RESULTS_DIR / f"{name}_{'int8' if quantize else 'fp32'}.tflite"
        path.write_bytes(blob)
        return len(blob) / 1024 ** 2
    except Exception as e:  # conversion is a bonus; never crash the run
        print(f"TFLite conversion skipped ({e})")
        return None


def host_latency_ms(model, n=200):
    """Single-image inference time on THIS machine (not the edge device) - relative comparison only."""
    x = np.random.rand(1, *IMG_SIZE, 3).astype("float32") * 255.0
    for _ in range(10):                      # warm-up
        model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x, training=False)
    return (time.perf_counter() - t0) / n * 1000.0



# Run the whole SOTA fine tuning and evaluation

all_results = {}
for name in BACKBONES:
    model, history, boundary, epoch_times = train_two_phase(name)
    plot_curves(history, boundary, name)

    acc, prec, rec, f1 = evaluate_on_test(model, name)

    keras_path = RESULTS_DIR / f"{name}_final.keras"
    model.save(keras_path)
    total, trainable = count_params(model)

    all_results[name] = {
        "test_accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec), 4),
        "macro_recall": round(float(rec), 4),
        "macro_f1": round(float(f1), 4),
        "total_params": int(total),
        "trainable_params_after_finetune": trainable,
        "size_mb_keras_file": round(os.path.getsize(keras_path) / 1024 ** 2, 2),
        "size_mb_fp32_estimate": round(total * 4 / 1024 ** 2, 2),   # params x 4 bytes
        "size_mb_tflite_fp32": tflite_size_mb(model, name, quantize=False),
        "size_mb_tflite_int8": tflite_size_mb(model, name, quantize=True),
        "avg_time_per_epoch_s": round(float(np.mean(epoch_times)), 2),
        "epochs_run": len(epoch_times),
        "host_latency_ms_per_image": round(host_latency_ms(model), 2),
    }
    print(f"\n{name}: {json.dumps(all_results[name], indent=2)}")

(RESULTS_DIR / "sota_results.json").write_text(json.dumps(all_results, indent=2))

print("\n" + "=" * 78)
print(f"{'Model':<14}{'Params':>12}{'Size MB':>10}{'Test acc':>10}{'Prec':>8}{'Recall':>8}{'s/epoch':>9}")
for n, r in all_results.items():
    print(f"{n:<14}{r['total_params']:>12,}{r['size_mb_keras_file']:>10}"
          f"{r['test_accuracy']:>10}{r['macro_precision']:>8}{r['macro_recall']:>8}"
          f"{r['avg_time_per_epoch_s']:>9}")