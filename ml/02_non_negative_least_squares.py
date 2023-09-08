"""
Non-Negative Least Squares

Ref:
    https://scikit-learn.org/stable/auto_examples/linear_model/plot_nnls.html#sphx-glr-auto-examples-linear-model-plot-nnls-py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# generate some random data
np.random.seed(42)
n_samples, n_features = 200, 50
X = np.random.randn(n_samples, n_features)
true_coef = 3 * np.random.randn(n_features)
# threshold coefficients to render them non-negative
true_coef[true_coef < 0] = 0
y = np.dot(X, true_coef)
# add some noise
y += 5 * np.random.normal(size=(n_samples,))

# split the data in train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# fit the non-negative least squares
reg_nnls = LinearRegression(positive=True)
y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
r2_score_nnls = r2_score(y_test, y_pred_nnls)
print(f'NNLS R2 score: {r2_score_nnls:.3f}')

# fit an ols
reg_ols = LinearRegression()
y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
r2_score_ols = r2_score(y_test, y_pred_ols)
print(f'OLS R2 score: {r2_score_ols:.3f}')

# plot
fig, ax = plt.subplots()
ax.plot(reg_ols.coef_, reg_nnls.coef_, '.')
low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()
low = max(low_x, low_y)
high = min(high_x, high_y)
ax.plot([low, high], [low, high], 'r--')
ax.set_xlabel('OLS regression coefficients')
ax.set_ylabel('NNLS regression coefficients')
plt.show()
