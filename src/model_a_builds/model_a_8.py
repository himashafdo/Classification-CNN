from tensorflow.keras import layers, models, losses

model_a = models.Sequential([

    layers.Input((64, 64, 3)),

    
    layers.RandomFlip('horizontal_and_vertical'),  # to reduce overfitting
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
    layers.Rescaling(1./255),

    
    layers.Conv2D(32, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    
    layers.Conv2D(64, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    
    layers.Conv2D(128, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    
    layers.Conv2D(256, (3,3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool2D((2,2)),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(9)   

])

model_a.compile(
    optimizer='adam',
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model_a.summary()