"""
For the model A I was thinking to use 3 conv-pooling layers with 2x2 pooling.
And filter amount will be 16, 32, 64 in each successive block, with each being a 
3x3 filter.
"""

from tensorflow.keras import layers, models


model_a = models.Sequential([

    #First convolutional layer with 16 filters and a 3x3 kernel
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3), padding='same'),
    #Maxpooling layer no.1 
    layers.MaxPool2D((2,2)),

    #Second convolutional layer with 32 filters and a 3x3 kernel
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    #Maxpooling layer no.2 
    layers.MaxPool2D((2,2)),

    #Third convolutional layer with 64 filters and a 3x3 kernel
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    #Maxpooling layer no.3
    layers.MaxPool2D((2,2)),

    #flattening to a 1D array 
    layers.GlobalAveragePooling2D(),

    #added another hidden dense layer to see the reaction
    layers.Dense(128, activation='relu'),
    
    

    layers.Dense(9, activation='softmax')

])

model_a.summary()