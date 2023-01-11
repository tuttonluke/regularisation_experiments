# %%
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
import numpy as np
import matplotlib.pyplot as plt

# %%
def create_dummy_data():
    X, y = make_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, 
                                                                    y_test, 
                                                                    test_size=0.3)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_val = ss.transform(X_validation)
    
    return X_train, X_val, y_train, y_validation

def get_loss_curves(model, X_train, X_val, y_train, y_val):
    
    train_errors = []
    val_errors = []
    n_epochs = 100

    for epoch in range(n_epochs):
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        train_errors.append(mean_squared_error(y_train, y_train_pred))

        y_val_pred = model.predict(X_val)
        val_errors.append(mean_squared_error(y_val, y_val_pred))

    best_epoch = np.argmin(val_errors)
    best_val_rmse = np.sqrt(val_errors[best_epoch])
    print(f"Best Epoch: {best_epoch}\nBest Validation Loss: {best_val_rmse}")

    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation Set")
    plt.plot(np.sqrt(train_errors), "r--", linewidth=3, label="Training Set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
#%%
np.random.seed(42)
X_train, X_val, y_train, y_val = create_dummy_data()
sgd_regressor = SGDRegressor(max_iter=1, tol=None, warm_start=True)
get_loss_curves(sgd_regressor, X_train, X_val, y_train, y_val)