
import pandas as pd
import numpy as np


#Get data
master_data = pd.read_csv("./footlength_height/footlength_height.csv")
x = master_data.iloc[:, 0].values.reshape(-1, 1)
y = master_data.iloc[:, 1].values



#Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
#x_train = scaler.fit_transform(x_train)
#x_test = scaler.fit_transform(x_test)

#split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Do regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()


lin_reg.fit(x_train, y_train)
y_preds = lin_reg.predict(x_test)


#plot
from matplotlib import pyplot as plt
plt.scatter(x, y, color="blue")
plt.plot(x, lin_reg.predict(x), color="red", linewidth=4)
plt.show()
print(True)


from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_preds))