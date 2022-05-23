import pandas as pd
import numpy as np
import os
from pandas.api.types import is_object_dtype, is_numeric_dtype

#import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


#Get dataset

master_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))

cols_to_drop = []
x_orig = None

def clean_col(col):
    "Cleans one column from master_df."
    global master_df
    global x_orig
    


    raw_data = master_df.loc[:, col]
    

    #If it is a string, then just put the mode in there
    if is_object_dtype(master_df[col]):
        replacement = raw_data.mode()
        #Else, if it is a median (meaning numerical), set the values to the median.
    elif is_numeric_dtype(master_df[col]):
        replacement = raw_data.median()
    else:
        raise ValueError("Not an str, nor a number")


    #If it is a string: 
    if is_object_dtype(master_df[col]):

        #OHE, store names in list,
        OHE_df = pd.get_dummies(master_df[col], prefix=col)
        cols_to_drop.append(col)
        #Append to master_df.
        x_orig = pd.concat([x_orig, OHE_df], axis=1)


    #Replace the values
    master_df.loc[:, col].replace(["NA", np.NAN], [replacement, replacement], inplace=True)



x_column_indexes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
x_orig = master_df.iloc[:, x_column_indexes]
#Loop and clean the data
for col in master_df.columns[x_column_indexes]:
    clean_col(col)



#Now, remove any string columns from x
x_orig.drop(cols_to_drop, axis=1, inplace=True)

print(True)