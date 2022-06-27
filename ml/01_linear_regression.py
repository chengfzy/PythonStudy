"""
Linear Regression Example

Ref: 
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# load the diabetes dateset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# split the data into training/test set
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# create linear regression object
regr = linear_model.LinearRegression()
# train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)
# make predictions using the test set
diabetes_y_pred = regr.predict(diabetes_X_test)

# print the coefficients
print(f'coefficients = {regr.coef_}')
# the mean squared error
print(f'mean squared error: {mean_squared_error(diabetes_y_test, diabetes_y_pred):.3f}')
# the coefficient of determination: 1 is perfect prediction
print(f'coefficient of determination: {r2_score(diabetes_y_test, diabetes_y_pred):.3f}')

# plot
plt.plot(diabetes_X_test, diabetes_y_test, 'b.')
plt.plot(diabetes_X_test, diabetes_y_pred, 'r-')
plt.xticks()
plt.yticks()
plt.show()