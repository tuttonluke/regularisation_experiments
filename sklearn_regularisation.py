# %%
from sklearn import datasets
from sklearn.model_selection import train_test_split
# %%
X, y = datasets.fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.3)