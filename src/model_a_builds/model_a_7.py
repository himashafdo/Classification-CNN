"""
Model A version 7 - this is the final version for model A, with the following changes:
- Added Batch Normalization after each Conv2D layer to stabilize and accelerate training.
but this model requires about 617K parameters, so we did this for purely to see the limit.
"""


from tensorflow.keras import layers, models

model_a = models.Sequential([

    layers.Input((64, 64, 3)),
    layers.Rescaling(1./255),

    #Block 1: 32 filters
    layers.Conv2D(32, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2)),

    #Block 2: 64 filters
    layers.Conv2D(64, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2)),

    #Block 3: 128 filters
    layers.Conv2D(128, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2)),

    #Block 4: 256 filters
    layers.Conv2D(256, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2)),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(9, activation='softmax')

])

model_a.summary()