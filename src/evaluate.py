"""
src/evaluate.py
Evaluates Model A on the test dataset, computes precision, recall, accuracy,
and plots the confusion matrix.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# 1. Paths and Settings
TEST_DATA_DIR = Path("data/RealWaste_split/test") 

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
MODEL_PATH = Path("results/model_a_version_1.keras")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading test dataset...")
test_ds = image_dataset_from_directory(
    TEST_DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False 
)

class_names = test_ds.class_names
print(f"Classes found: {class_names}")

# 2. Load the trained model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# 3. Generate predictions across the test set
print("Generating predictions on the test set...")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 4. Calculate and print metrics (Precision, Recall, Accuracy)
print("\n--- Classification Report ---")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Save text report
report_path = RESULTS_DIR / "model_a_classification_report.txt"
report_path.write_text(report)
print(f"Classification report saved to {report_path}")

# 5. Generate and plot Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Model A - Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right')
plt.yticks(tick_marks, class_names)

# Add text annotations inside the confusion matrix cells
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

cm_path = RESULTS_DIR / "model_a_confusion_matrix.png"
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_path}")