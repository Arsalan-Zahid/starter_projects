#FIRST OWN ML PROJECT!!! 4/22/22

import numpy as np
import pandas as pd

#import sklearn stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot 


#load data
master = pd.read_csv("./breast_cancer_data.csv")
x_original = master.iloc[:, 2:29]
y_original = master.iloc[:, 1]


#Split
x_train, x_test, y_train, y_test = train_test_split(x_original, y_original, test_size=0.5)

scaler = StandardScaler()

#Scale
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

"""
#logistic regression
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression()
logistic_classifier.fit(x_train, y_train)


#predict
y_preds = logistic_classifier.predict(x_test)


print("conf matrix for logistic classifier", confusion_matrix(y_test, y_preds))
"""
#TRY FOR SVC

from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)

y_preds = svm.predict(x_test)
print("conf matrix for SVC", confusion_matrix(y_test, y_preds))


print(True)