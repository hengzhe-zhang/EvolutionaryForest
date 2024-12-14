import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from evolutionary_forest.model.OptimalKNN import OptimalKNN


class DynamicSelectionOptimalKNN(BaseEstimator, RegressorMixin):
    def __init__(self, des_neighbours=None, criteria="density", **knn_params):
        """
        Gradient Boosting Regressor using Linear Regression and OptimalKNN.

        Parameters:
        - knn_params: dict, Parameters for OptimalKNN.
        """
        self.knn_params = knn_params if knn_params is not None else {}
        self.des_neighbours = des_neighbours
        self.criteria = criteria
        self.dt = DecisionTreeClassifier(max_depth=1)

    def calculate_criteria(self, criteria, errors, distances, indices):
        """
        Compute criteria values based on the provided method.

        Parameters:
        - criteria: str, Criteria type ('variance', 'density', 'mean_error', etc.).
        - errors: array, Errors for the model.
        - distances: array, Distances to nearest neighbors.
        - indices: array, Indices of nearest neighbors.

        Returns:
        - array, Criteria values.
        """
        if criteria == "variance":
            return np.var(errors[indices], axis=1)
        elif criteria == "density":
            return 1 / (np.mean(distances, axis=1) + 1e-8)
        elif criteria == "mean_error":
            return np.mean(errors[indices], axis=1)
        elif criteria == "max_error":
            return np.max(errors[indices], axis=1)
        else:
            raise ValueError(f"Unknown criteria: {criteria}")

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

        # Find nearest neighbors
        distances, indices = self.knn_model_.knn.kneighbors(
            X, n_neighbors=self.des_neighbours
        )

        # Compute criteria for nearest neighbors
        near_knn_error = self.calculate_criteria(
            self.criteria, self.knn_error, distances, indices
        )

        # Determine which model performs better for each sample
        label = np.argmin(np.vstack((self.lr_error, self.knn_error)).T, axis=1)

        # Train the decision tree on near_knn_error and labels
        self.dt.fit(near_knn_error.reshape(-1, 1), label)
        # plot_tree(self.dt)
        # plt.show()
        return self

    def predict(self, X):
        """Predict using the gradient boosting model."""
        check_is_fitted(self, ["linear_model_", "knn_model_"])
        X = check_array(X)

        # Find nearest neighbors using the KNN model
        distances, indices = self.knn_model_.knn.kneighbors(
            X, n_neighbors=self.des_neighbours
        )

        # Compute criteria for nearest neighbors
        near_knn_error = self.calculate_criteria(
            self.criteria, self.knn_error, distances, indices
        )

        # Use the decision tree to select the model for each sample
        select = self.dt.predict(near_knn_error.reshape(-1, 1)).astype(bool)

        final_prediction = np.zeros(X.shape[0])

        # Use Linear Regression for selected samples
        if np.any(select):
            final_prediction[select] = self.linear_model_.predict(X[select])

        # Use KNN for the remaining samples
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

    # Initialize the gradient boosting model
    for c in ["variance", "density", "mean_error", "max_error"]:
        model = DynamicSelectionOptimalKNN(
            des_neighbours=5,
            criteria=c,
        )

        # Fit and evaluate
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(c)
        print("Training R2 Score:", r2_score(y_train, model.predict(X_train)))
        print("R2 Score:", r2_score(y_test, predictions))

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
