import tensorflow as tf
import pandas as pd
import numpy as np

master_df = pd.read_csv("./breast_cancer/breast_cancer_data.csv")

#Change strings to integers

index_of_m = master_df.where(master_df["diagnosis"] == "M").dropna(subset=["diagnosis"]).index
index_of_b = master_df.where(master_df["diagnosis"] == "B").dropna(subset=["diagnosis"]).index

master_df.iloc[index_of_b, 1] = 0
master_df.iloc[index_of_m, 1] = 1

#Get data
x = master_df.iloc[:, 2:29]
y = master_df.iloc[:, 1]

x = np.asarray(x).astype('float32')
y = np.asarray(y).astype('float32')

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#Get ANN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape=(x_train.shape[1],), activation="sigmoid"))
model.add(tf.keras.layers.Dense(32, activation="sigmoid"))
model.add(tf.keras.layers.Dense(32, activation="sigmoid"))
model.add(tf.keras.layers.Dense(1,   activation="sigmoid"))

#Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(    monitor='accuracy',
    patience=100,
    mode='auto',
    restore_best_weights=True
)

#fit data
model.fit(x_train, y_train, epochs=1000, callbacks=[early_stopping])

y_preds = model.predict(x_test)
model.evaluate(x_test, y_test)