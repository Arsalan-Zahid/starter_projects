import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

#Get data
iris_csv = pd.read_csv("./iris/iris.csv")
X = iris_csv.iloc[:, :-1]
y = iris_csv.iloc[:, -1]

#split
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.4)

#preprocess
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit

features_num = X.columns
features_cat = ["species"]

preprocessor = make_column_transformer(
    (StandardScaler(), features_num)#,
#    (OneHotEncoder(), features_cat)
)

x_train = np.asarray(preprocessor.fit_transform(x_train))
x_val = np.asarray(preprocessor.transform(x_val))
preprocessor = OneHotEncoder(sparse=False)
y_train = preprocessor.fit_transform(y_train.values.reshape(-1,1))
y_val = preprocessor.transform(y_val.values.reshape(-1,1))


model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),

    layers.Dense(64*2, activation="relu"),
    layers.Dropout(0.5),
    layers.BatchNormalization(),

    layers.Dense(64*4, activation="relu"),
    layers.Dropout(0.5),
    layers.BatchNormalization(),

    layers.Dense(3, activation = tf.nn.softmax)


])

model.compile(optimizer="adam",
    loss="categorical_crossentropy",
    metrics="accuracy"
)


history = model.fit( x_train, y_train,
    epochs=60,
    batch_size = 75,
    validation_data = [x_val, y_val]

)

print(True)