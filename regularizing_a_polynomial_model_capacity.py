# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# %%
def add_square_column(array, p):
    new_array = np.append(array, array**p, axis=1)
    return new_array

def compute_label(x):
    return 2 + x -  0.6*x**2 + 0.1*x**3

#%% create label vector
x = np.linspace(0, 5, num=100)
y = compute_label(x)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x, y)

plt.show()

# %% create feature vector
np.random.seed(42)
design_matrix = np.random.rand(1000, 1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)

poly = PolynomialFeatures(degree=1, include_bias=False)

train_poly_features = poly.fit_transform(X_train.reshape(-1, 1))
test_poly_features = poly.fit_transform(X_test.reshape(-1, 1))
validation_poly_features = poly.fit_transform(X_validation.reshape(-1, 1))


poly_reg_model = LinearRegression()
poly_reg_model.fit(train_poly_features, y_train)


#%%
y_predicted_train = poly_reg_model.predict(train_poly_features)
y_predicted_test = poly_reg_model.predict(test_poly_features)
y_predicted_validation = poly_reg_model.predict(validation_poly_features)


train_loss = mean_squared_error(y_train, y_predicted_train)
test_loss = mean_squared_error(y_test, y_predicted_test)
validation_loss = mean_squared_error(y_validation, y_predicted_validation)
print(f"Train Loss: {train_loss}\nTest Loss: {test_loss}\nValidation Loss: {validation_loss}.")

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(x, y)
# ax1.plot(x, linear_fit)
ax1.scatter(X_train, y_predicted_train, linewidth=0.01, color="purple")
ax1.scatter(X_test, y_predicted_test, linewidth=0.01, color="green")
ax1.scatter(X_validation, y_predicted_validation, linewidth=0.01, color="red")


plt.show()
# %%
def plot_graphs(no_orders):
    np.random.seed(42)
    x = np.linspace(0, 5, num=100)
    y = compute_label(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)
    fig = plt.figure()
    color_list = ["r", "g", "m", "c", "k", "y"]

    for i in range(no_orders):
        poly = PolynomialFeatures(degree=i+1, include_bias=False)
        train_poly_features = poly.fit_transform(X_train.reshape(-1, 1))
        test_poly_features = poly.fit_transform(X_test.reshape(-1, 1))
        validation_poly_features = poly.fit_transform(X_validation.reshape(-1, 1))
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(train_poly_features, y_train)
        y_predicted_train = poly_reg_model.predict(train_poly_features)
        y_predicted_test = poly_reg_model.predict(test_poly_features)
        y_predicted_validation = poly_reg_model.predict(validation_poly_features)


        train_loss = mean_squared_error(y_train, y_predicted_train)
        test_loss = mean_squared_error(y_test, y_predicted_test)
        validation_loss = mean_squared_error(y_validation, y_predicted_validation)
        print(f"Polynomial of degree {i}.")
        print(f"Train Loss: {train_loss}\nTest Loss: {test_loss}\nValidation Loss: {validation_loss}.\n")

        ax1 = fig.add_subplot(no_orders, 1, i+1)
        ax1.plot(x, y)
        
        ax1.scatter(X_train, y_predicted_train, linewidth=0.01, color=color_list[i])
    plt.show()

plot_graphs(5)

