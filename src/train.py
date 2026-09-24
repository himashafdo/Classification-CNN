from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from model_a import model_a

TRAIN_VAL_DATA_DIR = Path("data/RealWaste_split")
IMG_SIZE = (64,64)
BATCH_SIZE = 64

train_ds = image_dataset_from_directory(
    TRAIN_VAL_DATA_DIR/ "train",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "int"
)

val_ds = image_dataset_from_directory(
    TRAIN_VAL_DATA_DIR/ "val",
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    label_mode = "int"
)

class_names = train_ds.class_names

train_labels = []
for _, labels in train_ds.unbatch():
    train_labels.append(labels.numpy())
train_labels = np.array(train_labels)

class_weights_array = compute_class_weight(
    class_weight="balanced", #since all classes doesn't have equal amount of imgs --> balanced iti 
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights_array))
print("Class weights:", class_weights)

model_a.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model_a.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 30,
    class_weight=class_weights
)
