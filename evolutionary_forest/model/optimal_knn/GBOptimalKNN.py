import random
import time

from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from evolutionary_forest.component.stgp.smooth_scaler import NearestValueTransformer
from evolutionary_forest.model.OptimalKNN import OptimalKNN

from sklearn.tree import DecisionTreeRegressor

from evolutionary_forest.model.linear_regression import BoundedRidgeRegressor


class RidgeBoostedKNN(BaseEstimator, RegressorMixin):
    def __init__(self, knn_params=None, bounded_ridge=False):
        """
        RidgeBoostedKNN: A two-stage regressor that first fits RidgeCV,
        then fits OptimalKNN on the residuals.

        Parameters
        ----------
        knn_params : dict or None
            Parameters passed to OptimalKNN.
        """
        self.knn_params = knn_params if knn_params is not None else {}
        self.time_information = {}
        self.bounded_ridge = bounded_ridge

    def fit(self, X, y):
        """Fit RidgeCV first, then OptimalKNN on residuals."""
        self.get_n_neighbors(y)

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Stage 1: Fit RidgeCV
        start = time.time()
        if self.bounded_ridge:
            self.ridge_model_ = BoundedRidgeRegressor()
        else:
            self.ridge_model_ = RidgeCV()
        self.ridge_model_.fit(X, y)
        end = time.time()
        self.time_information["Ridge"] = end - start

        # Stage 2: Fit OptimalKNN on residuals
        residuals = y - self.ridge_model_.predict(X)
        self.knn_model_ = OptimalKNN(**self.knn_params)
        self.knn_model_.fit(X, residuals)

        return self

    def get_n_neighbors(self, y):
        if self.knn_params.get("n_neighbors", "") == "Adaptive":
            if len(np.unique(y)) <= 50:
                self.knn_params["n_neighbors"] = 20
            else:
                self.knn_params["n_neighbors"] = 5

    def predict(self, X):
        """Predict by combining RidgeCV predictions + KNN residual correction."""
        check_is_fitted(self, ["ridge_model_", "knn_model_"])
        X = check_array(X)

        # Base prediction from RidgeCV
        predictions = self.ridge_model_.predict(X)

        # Add correction from OptimalKNN
        predictions += self.knn_model_.predict(X)

        return predictions


class RidgeBoostedDT(BaseEstimator, RegressorMixin):
    def __init__(self, leaf_size):
        self.leaf_size = leaf_size

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.ridge_model_ = RidgeCV()
        self.ridge_model_.fit(X, y)

        residuals = y - self.ridge_model_.predict(X)
        self.dt_model_ = DecisionTreeRegressor(min_samples_leaf=self.leaf_size)
        self.dt_model_.fit(X, residuals)

        return self

    def predict(self, X):
        ridge_pred = self.ridge_model_.predict(X)
        residual_pred = self.dt_model_.predict(X)
        return ridge_pred + residual_pred


class RidgeKNNTree(RidgeBoostedKNN):
    """
    RidgeKNNTree: A three-stage regressor.
      1. Fit RidgeCV to capture linear trends.
      2. Fit OptimalKNN on the residuals from RidgeCV.
      3. Fit DecisionTreeRegressor on the residuals from KNN.
    """

    def __init__(self, knn_params=None, tree_params=None):
        super().__init__(knn_params=knn_params)
        self.tree_params = tree_params if tree_params is not None else {}

    def fit(self, X, y):
        """Fit RidgeCV -> OptimalKNN -> DecisionTreeRegressor in sequence."""
        self.get_n_neighbors(y)
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Stage 1: RidgeCV
        self.ridge_model_ = RidgeCV()
        self.ridge_model_.fit(X, y)
        residuals_1 = y - self.ridge_model_.predict(X)

        # Stage 2: OptimalKNN
        self.knn_model_ = OptimalKNN(**self.knn_params)
        self.knn_model_.fit(X, residuals_1)
        residuals_2 = residuals_1 - self.knn_model_.predict(X)

        # Stage 3: Decision Tree on residuals
        self.tree_model_ = DecisionTreeRegressor(**self.tree_params)
        self.tree_model_.fit(X, residuals_2)

        return self

    def predict(self, X):
        """Combine predictions from all three stages."""
        X = check_array(X)
        ridge_pred = self.ridge_model_.predict(X)
        knn_pred = self.knn_model_.predict(X)
        tree_pred = self.tree_model_.predict(X)
        return ridge_pred + knn_pred + tree_pred


class RidgeBoostedSimpleKNN(RidgeBoostedKNN):
    """
    Variant of RidgeBoostedKNN that uses scikit-learn's
    KNeighborsRegressor instead of OptimalKNN.
    """

    def fit(self, X, y):
        """Fit RidgeCV first, then plain KNeighborsRegressor on residuals."""
        self.get_n_neighbors(y)

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Stage 1: RidgeCV
        self.ridge_model_ = RidgeCV()
        self.ridge_model_.fit(X, y)

        # Stage 2: Fit KNeighborsRegressor on residuals
        residuals = y - self.ridge_model_.predict(X)
        self.knn_model_ = KNeighborsRegressor(**self.knn_params)
        self.knn_model_.fit(X, residuals)

        return self


class ConstraintRidgeBoostedKNN(RidgeBoostedKNN):
    def fit(self, X, y):
        self.y_transformer = NearestValueTransformer()
        self.y_transformer.fit_transform(y)
        return super().fit(X, y)

    def predict(self, X):
        preds = super().predict(X)
        return self.y_transformer.transform(preds)


class RandomNeighborRidgeBoostedKNN(RidgeBoostedKNN):
    def __init__(self, knn_params=None, neighbor_choices=None):
        super().__init__(knn_params=knn_params)
        self.neighbor_choices = (
            neighbor_choices if neighbor_choices is not None else [5, 10, 20]
        )

        # Randomly select n_neighbors at initialization
        self.n_neighbors_ = random.choice(self.neighbor_choices)

        # Update knn_params with the randomly chosen n_neighbors
        if self.knn_params is None:
            self.knn_params = {}
        self.knn_params["n_neighbors"] = self.n_neighbors_

    def fit(self, X, y):
        """Fit RidgeCV first, then OptimalKNN with pre-selected n_neighbors on residuals."""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Stage 1: Fit RidgeCV
        self.ridge_model_ = RidgeCV()
        self.ridge_model_.fit(X, y)

        # Stage 2: Fit OptimalKNN on residuals (n_neighbors already set in __init__)
        residuals = y - self.ridge_model_.predict(X)
        self.knn_model_ = OptimalKNN(**self.knn_params)
        self.knn_model_.fit(X, residuals)

        return self


class SplitFeatureRidgeKNN(RidgeBoostedKNN):
    """
    A variant where the first half of features go to Ridge,
    and the second half go to KNN (which models residuals).

    Parameters
    ----------
    knn_params : dict or None
        Parameters passed to OptimalKNN.
    split_index : int or None
        Index to split features. If None, splits at midpoint.
    """

    def __init__(self, knn_params=None, split_index=None):
        super().__init__(knn_params=knn_params)
        self.split_index = split_index

    def fit(self, X, y):
        """Fit Ridge on first half of features, KNN on second half."""
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Determine split point
        if self.split_index is None:
            self.split_index_ = X.shape[1] // 2
        else:
            self.split_index_ = self.split_index

        # Validate split index
        if not (0 < self.split_index_ < X.shape[1]):
            raise ValueError(
                f"split_index must be between 0 and {X.shape[1]}, "
                f"got {self.split_index_}"
            )

        # Split features
        X_ridge = X[:, : self.split_index_]
        X_knn = X[:, self.split_index_ :]

        # Stage 1: Fit RidgeCV on first half of features
        self.ridge_model_ = RidgeCV()
        self.ridge_model_.fit(X_ridge, y)

        # Stage 2: Fit OptimalKNN on second half of features using residuals
        residuals = y - self.ridge_model_.predict(X_ridge)
        self.knn_model_ = OptimalKNN(**self.knn_params)
        self.knn_model_.fit(X_knn, residuals)

        return self

    def predict(self, X):
        """Predict by combining Ridge (first features) + KNN (second features)."""
        check_is_fitted(self, ["ridge_model_", "knn_model_", "split_index_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained on "
                f"{self.n_features_in_} features"
            )

        # Split features
        X_ridge = X[:, : self.split_index_]
        X_knn = X[:, self.split_index_ :]

        # Base prediction from Ridge (first half of features)
        predictions = self.ridge_model_.predict(X_ridge)

        # Add correction from KNN (second half of features)
        predictions += self.knn_model_.predict(X_knn)

        return predictions


class OptimalKNNRandomDT(RidgeBoostedKNN):
    def __init__(self, knn_params=None):
        """
        Gradient Boosting Regressor using OptimalKNN and Linear Regression.

        Parameters:
        - knn_params: dict, Parameters for OptimalKNN.
        """
        self.knn_params = knn_params if knn_params is not None else {}

    def fit(self, X, y):
        """Fit the gradient boosting model."""
        self.n_features_in_ = X.shape[1]
        X, y = check_X_y(X, y)
        self.get_n_neighbors(y)

        self.knn_model_ = OptimalKNN(**self.knn_params)
        self.knn_model_.fit(X, y)

        predictions, X_t = self.knn_model_.predict(X, return_transformed=True)
        residuals = y - predictions

        self.initial_model_ = DecisionTreeRegressor()
        self.initial_model_.fit(X_t, residuals)

        return self

    def predict(self, X):
        """Predict using the gradient boosting model."""
        check_is_fitted(self, ["initial_model_", "knn_model_"])
        X = check_array(X)

        predictions, X_t = self.knn_model_.predict(X, return_transformed=True)
        predictions += self.initial_model_.predict(X_t)
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
    model = OptimalKNNRandomDT(
        knn_params={"n_neighbors": 5, "distance": "SkipUniform"},
    )

    # Fit and evaluate
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Train R2:", r2_score(y_train, model.predict(X_train)))
    print("R2:", r2_score(y_test, predictions))
