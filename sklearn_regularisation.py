# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import typing
import itertools
import numpy as np
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

def k_fold(dataset, n_splits: int = 5):
    chunks = np.array_split(dataset, n_splits)
    for i in range(n_splits):
        training = chunks[:i] + chunks[i+1:]
        validation = chunks[i]
        yield np.concatenate(training), validation

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
    np.random.seed(42)
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


# %% k-fold validation and grid search
grid = {
    "alpha" : [0.1, 0.2, 0.3],
    "max_iter" : [None, 500, 1000, 2000]
}

np.random.seed(42)
n_splits = 5
X, y = datasets.fetch_california_housing(return_X_y=True)
model_name_list = ["Linear Regression", "Ridge", "Lasso"]
model_list = [LinearRegression(), Ridge(), Lasso()]

def model_k_fold_grid_search(model):
    best_hyperparams, best_loss = None, np.inf
    for hyperparams in grid_search(grid):
        loss = 0
        # instead of validation we use K-fold
        for (X_train, X_validation), (y_train, y_validation) in zip(
            k_fold(X, n_splits), k_fold(y, n_splits)
        ):
            
            model.fit(X_train, y_train)

            y_validation_pred = model.predict(X_validation)
            fold_loss = mean_squared_error(y_validation, y_validation_pred)
            loss += fold_loss
        # Take the mean of all the folds as the final validation score
        total_loss = loss / n_splits
        print(f"H-Params: {hyperparams}\nLoss: {total_loss}\n")
        if total_loss < best_loss:
            best_loss = total_loss
            best_hyperparams = hyperparams
    print(f"Best Loss: {best_loss}\nBest Hyperparameters: {best_hyperparams}\n")

for i, model in enumerate(model_list):
    print(model_name_list[i])
    model_k_fold_grid_search(model)