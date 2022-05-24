import pandas as pd
import numpy as np
import os
from pandas.api.types import is_object_dtype, is_numeric_dtype
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf

master_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))

#quickly, drop any values that don't have a y value
master_df.dropna(subset=["hospital_death"], inplace=True)


#Get original data
x_column_indexes = [4, 5, 6, 7, 8, 10, 11, 13, 14]
x_orig = master_df.iloc[:, x_column_indexes]
y_orig = master_df.iloc[:, 3]



from sklearn.impute import SimpleImputer


object_cols = [col for col in x_orig.columns if is_object_dtype(x_orig[col])]
numeric_cols = [col for col in x_orig.columns if is_numeric_dtype(x_orig[col])]


from sklearn.compose import make_column_transformer



#split data
from sklearn.model_selection import train_test_split



transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"),
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, numeric_cols),
    (transformer_cat, object_cols),
)

# stratify - make sure classes are evenlly represented across splits
X_train, X_valid, y_train, y_valid = \
    train_test_split(x_orig, y_orig, stratify=y_orig, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]





#Build model
from tensorflow.keras import layers

model = tf.keras.models.Sequential([
    layers.BatchNormalization(input_shape=(X_train.shape[1],)),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(1, activation="sigmoid")

])

#Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"])

#Get early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
    monitor="binary_accuracy"
)

X_train = X_train.toarray()
X_valid = X_valid.toarray()

#fit the data
history = model.fit(X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=1000,
    epochs=400,
    callbacks = [early_stopping])

history_df = pd.DataFrame(history.history)

history_df.loc[:, ["loss", "val_loss"]].plot(title="cross-entropy")
history_df.loc[:, ["binary_accuracy", "val_binary_accuracy"]].plot(title="accuracy")

from matplotlib.pyplot import show
show()