# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# %%
def add_square_column(array, p):
    new_array = np.append(array, array**p, axis=1)
    return new_array

def compute_label(x):
    return 2 + x -  0.6*x**2 + 0.1*x**3

#%% create label vector
x = np.linspace(0, 5, num=1000)
y = compute_label(x)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x, y)

plt.show()

# %% create feature vector
np.random.seed(42)
design_matrix = np.random.rand(1000, 1)

# X_train, X_test, y_train, y_test = train_test_split(design_matrix, y, test_size=0.3)
# X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)

# linear_model = LinearRegression()
# linear_model.fit(x, y_train)

poly = PolynomialFeatures(degree=3, include_bias=False)
poly_features = poly.fit_transform(x.reshape(-1, 1))
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, y)

#%%
# linear_fit = linear_model.intercept_ - x*linear_model.coef_
y_predicted = poly_reg_model.predict(poly_features)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x, y)
# ax1.plot(x, linear_fit)
ax1.plot(x, y_predicted, color="purple")

plt.show()
# %%
