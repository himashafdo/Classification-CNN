from tensorflow.keras import layers, models

def block(filters):
    return [
        layers.Conv2D(filters, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(filters, 3, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPool2D(2),
    ]

model_a = models.Sequential([
    layers.Input((64,64,3)),
    layers.Rescaling(1./255),
    *block(32), *block(64), *block(128),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(9, activation='softmax'),
])

model_a.summary()