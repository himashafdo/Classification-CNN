"""
Model B version 2 - lightweight CNN using depthwise separable convolutions.

This version mirrors Model A version 8 layer for layer (same augmentation,
same 32 -> 64 -> 128 -> 256 filter progression, same BN + ReLU + MaxPool
blocks, same classifier head). The ONLY change is that Conv2D in blocks 2-4
is replaced with SeparableConv2D, so any difference between A and B comes
from the convolution type alone.

Changes from model_b_1:
- Added Rescaling(1./255) - model_b_1 was feeding raw 0-255 pixels.
- Replaced Flatten with GlobalAveragePooling2D - in model_b_1 the
  Flatten -> Dense(9) layer alone used 36,873 of the 40,052 params.
- Added BatchNormalization after each conv (use_bias=False since BN's
  beta already acts as a bias).
- Output is logits (no softmax) - argmax of logits gives the same class,
  and it avoids exp() on the edge device.

Block 1 is kept as a standard Conv2D (stem), same as MobileNetV1:
a depthwise filter on only 3 input channels (RGB) learns very little,
and this layer is only 864 weights.
"""

from tensorflow.keras import layers, models, losses

model_b = models.Sequential([

    layers.Input((64, 64, 3)),

    
    layers.RandomFlip('horizontal_and_vertical'),  # to reduce overfitting
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.Rescaling(1./255),

    #Block 1: stem, standard conv
    layers.Conv2D(32, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    #Block 2: depthwise separable 32 -> 64
    layers.SeparableConv2D(64, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    #Block 3: depthwise separable 64 -> 128
    layers.SeparableConv2D(128, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    #Block 4: depthwise separable 128 -> 256
    layers.SeparableConv2D(256, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(9)   

])

model_b.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model_b.summary()