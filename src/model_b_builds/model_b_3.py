"""

Model B version 3 - depthwise separable version of Model A version 7
(the final Model A).

Same structure as model_a_7: 4 blocks (32, 64, 128, 256 filters), two conv
layers per block in blocks 1-3, BN + ReLU after every conv, no data
augmentation, softmax output. The changes are:
- every Conv2D except the very first one (stem) is replaced with
  SeparableConv2D
- Dense head reduced from 128 to 96 units, because the exact mirror of
  model_a_7 with Dense(128) came to 104,873 params (over the 100k limit)

The first Conv2D is kept standard, same as MobileNetV1: a depthwise filter on
only 3 input channels (RGB) learns very little, and this layer is only 864
weights.

Model B v2 (mirror of model_a_8) only reached 57.4% test accuracy, with the
model over-predicting Vegetation / Food Organics. v3 follows model_a_7, which
reached 79% with deeper blocks and no augmentation.
"""


from tensorflow.keras import layers, models

model_b = models.Sequential([

    layers.Input((64, 64, 3)),
    layers.Rescaling(1./255),

    #Block 1: 32 filters (first conv standard = stem)
    layers.Conv2D(32, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.SeparableConv2D(32, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2)),

    #Block 2: 64 filters
    layers.SeparableConv2D(64, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.SeparableConv2D(64, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2)),

    #Block 3: 128 filters
    layers.SeparableConv2D(128, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.SeparableConv2D(128, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2)),

    #Block 4: 256 filters
    layers.SeparableConv2D(256, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D((2,2))])