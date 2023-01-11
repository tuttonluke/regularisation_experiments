# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import typing
import itertools
# %%
def min_max_norm(X, y):
    sX = preprocessing.MinMaxScaler()
    sy = preprocessing.MinMaxScaler()

    scaled_X = sX.fit_transform(X)
    scaled_y = sy.fit_transform(y.reshape(-1, 1))

    return scaled_X, scaled_y

def grid_search(hyperparameters: typing.Dict[str, typing.Iterable]):
    keys, values = zip(*hyperparameters.items())
    yield from (dict(zip(keys, v)) for v in itertools.product(*values))

# %%
X, y = datasets.fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)

# min-max normalise data
X_train_scaled, y_train_scaled = min_max_norm(X_train, y_train)
X_test_scaled, y_test_scaled = min_max_norm(X_test, y_test)
X_validation_scaled, y_validation_scaled = min_max_norm(X_validation, y_validation)

# %% unregularised model
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train_scaled)

y_predicted_train = model.predict(X_train_scaled)
y_predicted_test = model.predict(X_test_scaled)
y_predicted_validation = model.predict(X_validation_scaled)

train_loss = mean_squared_error(y_train, y_predicted_train)
test_loss = mean_squared_error(y_test, y_predicted_test)
validation_loss = mean_squared_error(y_validation, y_predicted_validation)

print(f"Train Loss: {train_loss}\nTest Loss: {test_loss}\nValidation Loss: {validation_loss}.")

# %%
def model_data(model):
    X, y = datasets.fetch_california_housing(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)

    # min-max normalise data
    X_train_scaled, y_train_scaled = min_max_norm(X_train, y_train)
    X_test_scaled, y_test_scaled = min_max_norm(X_test, y_test)
    X_validation_scaled, y_validation_scaled = min_max_norm(X_validation, y_validation)
    
    model.fit(X_train_scaled, y_train_scaled)
    y_predicted_train = model.predict(X_train_scaled)
    y_predicted_test = model.predict(X_test_scaled)
    y_predicted_validation = model.predict(X_validation_scaled)

    train_loss = mean_squared_error(y_train_scaled, y_predicted_train)
    test_loss = mean_squared_error(y_test_scaled, y_predicted_test)
    validation_loss = mean_squared_error(y_validation_scaled, y_predicted_validation)
    print(f"Train Loss: {train_loss}\nTest Loss: {test_loss}\nValidation Loss: {validation_loss}.\n")


model_name_list = ["Linear Regression", "Ridge", "Lasso"]
model_list = [LinearRegression(), Ridge(), Lasso()]

for i, model in enumerate(model_list):
    print(model_name_list[i])
    model_data(model)
    
