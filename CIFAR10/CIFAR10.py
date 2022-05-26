import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd


"""
This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories.

"""

#Load the data
(training_images, training_labels), (val_images, val_labels)  = keras.datasets.cifar10.load_data()

#Normalize

training_images = training_images / 255.0
val_images = val_images / 255.0


INPUT_SHAPE = (32, 32, 3)

#Build the model
model = keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=INPUT_SHAPE),
    layers.MaxPool2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPool2D(),


    #Flatten and load
    layers.Flatten(),

    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),


    layers.Dense(10, activation=tf.nn.softmax)
])

#compile

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics="accuracy"

)

#Make the early stopping

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.01,
    patience=5,
    restore_best_weights=True

)

#Train the model
history = model.fit(
    training_images, training_labels, 
    validation_data=(val_images, val_labels),
    epochs=50,
    batch_size=1000,
    callbacks=[early_stopping]
)

#plot it
history_df = pd.DataFrame(history.history)
history_df.loc[:, ["loss", "val_loss", "accuracy", "val_accuracy"]].plot()

from matplotlib.pyplot import plot
plot()