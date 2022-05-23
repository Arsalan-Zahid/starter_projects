import pandas as pd
import numpy as np
import os
from pandas.api.types import is_object_dtype, is_numeric_dtype

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


#Get dataset

master_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))

#quickly, drop any values that don't have a y value
master_df.dropna(subset=["hospital_death"], inplace=True)

#FIRST, CLEAN THE DF AND OHE


cols_to_drop = []
x_orig = None

def clean_col(col):
    "Cleans one column from master_df."
    
    global x_orig
    


    raw_data = x_orig.loc[:, col]
    

    #If it is a string, then just put the mode in there
    if is_object_dtype(x_orig[col]):
        replacement = raw_data.mode()
        #Else, if it is a median (meaning numerical), set the values to the median.
    elif is_numeric_dtype(x_orig[col]):
        replacement = raw_data.median()
    else:
        raise ValueError("Not an str, nor a number")


    #If it is a string: 
    if is_object_dtype(x_orig[col]):

        #OHE, store names in list,
        OHE_df = pd.get_dummies(x_orig[col], prefix=col)
        cols_to_drop.append(col)
        #Append to master_df.
        x_orig = pd.concat([x_orig, OHE_df], axis=1)


    #Replace the values
    x_orig.loc[:, col].replace(["NA", np.NAN], [replacement, replacement], inplace=True)



x_column_indexes = [4, 7, 8, 10, 11, 13, 14]
x_orig = master_df.iloc[:, x_column_indexes]
#Loop and clean the data
for col in master_df.columns[x_column_indexes]:
    clean_col(col)



#Now, remove any string columns from x
x_orig.drop(cols_to_drop, axis=1, inplace=True)
y_orig = master_df.loc[:, "hospital_death"]#.drop(["index"], axis=1)


#___________________________________________________________________________________________________________________
#x_orig = np.asarray(x_orig).astype('float32')
#y_orig = np.asarray(y_orig).astype('float32')

feature_cols = []


bucketized_feature_column = tf.feature_column.bucketized_column(tf.feature_column.numeric_column("age"), boundaries=[30, 40, 50, 60, 70, 80, 90])
feature_cols.append(bucketized_feature_column)





#Now, split the data
#print(x_orig.to_dict())

x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=0.2)

#As array
"""
x_train = np.asarray(x_train).astype('float32')
y_train = np.asarray(y_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
y_test = np.asarray(y_test).astype('float32')"""

feature_layer = tf.keras.layers.DenseFeatures(feature_cols)

#Get the model and start adding layers
model = tf.keras.models.Sequential()
model.add(feature_layer)
model.add(tf.keras.layers.Dense(128, activation="sigmoid"))
model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

#compile
assert not np.any(np.isnan(x_train))
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])

columns = x_train.columns.to_list()

features = tf.io.parse_example(
    ..., features=tf.feature_column.make_parse_example_spec(columns))

model.fit(feature_layer(features), feature_layer(features), epochs = 100, batch_size=10000)
model.evaluate(x_test, y_test)
print(True)