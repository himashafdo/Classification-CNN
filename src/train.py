from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import callbacks
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from model_a_builds.model_a_7 import model_a

TRAIN_VAL_DATA_DIR = Path("data/RealWaste_split")
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

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

model_a.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cbs = [
    callbacks.ModelCheckpoint("results/model_a_version_7.keras",
                              monitor="val_accuracy", save_best_only=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    callbacks.EarlyStopping(patience=8, restore_best_weights=True),
]

history = model_a.fit(
    train_ds, validation_data=val_ds,
    epochs=50, class_weight=class_weights, callbacks=cbs
)


model_a.summary()

