from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from evolutionary_forest.model.OptimalKNN import OptimalKNN


class GradientBoostingWithOptimalKNN(BaseEstimator, RegressorMixin):
    def __init__(self, knn_params=None):
        """
        Gradient Boosting Regressor using OptimalKNN and Linear Regression.

        Parameters:
        - knn_params: dict, Parameters for OptimalKNN.
        """
        self.knn_params = knn_params if knn_params is not None else {}

    def fit(self, X, y):
        """Fit the gradient boosting model."""
        X, y = check_X_y(X, y)

        # Step 1: Fit the initial model (OptimalKNN)
        self.knn_model_ = OptimalKNN(**self.knn_params)
        self.knn_model_.fit(X, y)

        # Store residuals from the KNN model
        residuals = y - self.knn_model_.predict(X)

        # Step 2: Fit Linear Regression on residuals
        self.initial_model_ = RidgeCV()
        self.initial_model_.fit(X, residuals)

        return self

    def predict(self, X):
        """Predict using the gradient boosting model."""
        check_is_fitted(self, ["initial_model_", "knn_model_"])
        X = check_array(X)

        # Initial prediction from the KNN model
        predictions = self.knn_model_.predict(X)

        # Add contributions from the Linear Regression model
        predictions += self.initial_model_.predict(X)

        return predictions


# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    # Generate synthetic data
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
    model = GradientBoostingWithOptimalKNN(
        knn_params={"n_neighbors": 5, "distance": "SkipUniform"},
    )

    # Fit and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Train R2:", r2_score(y_train, model.predict(X_train)))
    print("R2:", r2_score(y_test, predictions))
