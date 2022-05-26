import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd

#Load data
(training_images, training_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#normalize
training_images = training_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation=tf.nn.softmax)
])

#Make early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=15,
    restore_best_weights=True
)

#Compile then fit
model.compile(optimizer="adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training_images, training_labels, 
batch_size=1000, 
epochs=400, 
callbacks=[early_stopping], 
validation_data=(test_images, test_labels))