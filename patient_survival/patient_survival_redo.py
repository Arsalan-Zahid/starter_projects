import pandas as pd
import numpy as np
import os
from pandas.api.types import is_object_dtype, is_numeric_dtype

import tensorflow as tf

master_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataset.csv"))

#quickly, drop any values that don't have a y value
master_df.dropna(subset=["hospital_death"], inplace=True)


#Get original data
x_column_indexes = [4, 5, 6, 7, 8, 10, 11, 13, 14]
x_orig = master_df.iloc[:, x_column_indexes]
y_orig = master_df.iloc[:, 3]



from sklearn.impute import SimpleImputer

#Loop through each column: 
for col in x_orig.columns:

    amount_of_null = x_orig[col].isnull().sum()

    #if 30% or more of the data is null, drop the col.
    if amount_of_null/len(x_orig) > 0.3:
        x_orig.drop([col,], axis=1, inplace=True)
    #Else, impute values.
    else:

        imputer = SimpleImputer(strategy="most_frequent")
        x_orig.loc[:, col] = imputer.fit_transform(np.asarray(x_orig.loc[:, col]).reshape(-1,1))




#Now, OHE
from sklearn.preprocessing import OneHotEncoder

#split data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_orig, y_orig, test_size=0.2)


