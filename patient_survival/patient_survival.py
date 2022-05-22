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



def clean_col(col):
    "Cleans one column from master_df."
    global master_df


    raw_data = master_df.loc[:, col]
    dtype = master_df.dtypes[col]

    #If it is a string, then just put the mode in theree
    if is_object_dtype(master_df[col]):
        replacement = raw_data.mode()
        #Else, if it is a median (meaning numerical), set the values to the median.
    elif is_numeric_dtype(master_df[col]):
        replacement = raw_data.median()
    else:
        raise ValueError("Not an str, nor a number")


    #Replace the values
    master_df.loc[:, col].replace(["NA", np.NAN], [replacement, replacement], inplace=True)



x_column_indexes = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

#Loop and clean the data
for x in master_df.columns[x_column_indexes]:
    master_df = clean_col(x)

