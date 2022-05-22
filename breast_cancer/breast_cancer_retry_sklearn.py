#Trying to do it again; 5/22/22

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#get data

master_df = pd.read_csv("./breast_cancer_data.csv")
x_original = master_df.iloc[:, 2:29]
y_original = master_df.iloc[:, 1]

#split data

x_train, x_test, y_train, y_test = train_test_split(x_original, y_original, test_size=0.2)

#Scale it
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#Now, fit it
svc_model = SVC()
svc_model.fit(x_train, y_train)

y_preds = svc_model.predict(x_test)

print(confusion_matrix(y_test, y_preds))