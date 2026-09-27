
from tensorflow.keras import layers, models


model_b = models.Sequential([

    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),

    layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same', input_shape=(64,64,3)),
    layers.MaxPool2D((2,2)),

    layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2,2)),

    layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPool2D((2,2)),

    layers.Flatten(),

    layers.Dense(9, activation='softmax')
])

model_b.summary()

"""

Upon the first iteration I noticed that there was severe overfitting
due to our datasets' small size. So we used Data augmentation methods in the training.
"""