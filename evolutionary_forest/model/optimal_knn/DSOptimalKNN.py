import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from evolutionary_forest.model.OptimalKNN import OptimalKNN


class DynamicSelectionOptimalKNN(BaseEstimator, RegressorMixin):
    def __init__(self, knn_params=None):
        """
        Gradient Boosting Regressor using Linear Regression and OptimalKNN.

        Parameters:
        - knn_params: dict, Parameters for OptimalKNN.
        """
        self.knn_params = knn_params if knn_params is not None else {}

    def fit(self, X, y):
        """Fit the gradient boosting model."""
        X, y = check_X_y(X, y)

        # Step 1: Fit the initial model (Linear Regression)
        self.linear_model_ = RidgeCV()
        self.linear_model_.fit(X, y)

        # Step 2: Fit OptimalKNN
        self.knn_model_ = OptimalKNN(**self.knn_params)
        self.knn_model_.fit(X, y)

        # Calculate residuals and errors
        lr_predictions = self.linear_model_.predict(X)
        knn_predictions = self.knn_model_.predict(X)

        self.lr_error = (y - lr_predictions) ** 2
        self.knn_error = (y - knn_predictions) ** 2

        return self

    def predict(self, X):
        """Predict using the gradient boosting model."""
        check_is_fitted(self, ["linear_model_", "knn_model_"])
        X = check_array(X)

        # Find nearest neighbors using the KNN model
        _, indices = self.knn_model_.knn.kneighbors(X)

        # Calculate average errors for the neighbors
        near_lr_error = np.mean(self.lr_error[indices], axis=1)
        near_knn_error = np.mean(self.knn_error[indices], axis=1)

        # Decide which model to use for each sample
        select = near_lr_error < near_knn_error

        final_prediction = np.zeros(X.shape[0])

        # Initial prediction from the linear model
        if np.any(select):
            final_prediction[select] = self.linear_model_.predict(X[select])

        # Add contributions from the KNN model
        if np.any(~select):
            final_prediction[~select] = self.knn_model_.predict(X[~select])

        return final_prediction


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_friedman1
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    # Load data
    # X, y = load_diabetes(return_X_y=True)
    X, y = make_friedman1(n_samples=1000, n_features=10, noise=0.1, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Compare with pure Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    print("Linear Regression Train R2:", r2_score(y_train, lr_model.predict(X_train)))
    print("Linear Regression R2:", r2_score(y_test, lr_predictions))

    # KNN
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    print("KNN Train R2:", r2_score(y_train, knn_model.predict(X_train)))
    print("KNN R2:", r2_score(y_test, knn_predictions))

    # Initialize the gradient boosting model
    model = DynamicSelectionOptimalKNN(
        knn_params={"n_neighbors": 5, "distance": "SkipUniform"},
    )

    # Fit and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Training R2 Score:", r2_score(y_train, model.predict(X_train)))
    print("R2 Score:", r2_score(y_test, predictions))
