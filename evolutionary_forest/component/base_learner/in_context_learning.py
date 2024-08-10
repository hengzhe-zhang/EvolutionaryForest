import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge


class InContextLearnerRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, k=3, nn_dimensions=1, use_average=True, **param):
        self.k = k
        self.nn_dimensions = nn_dimensions
        self.use_average = use_average
        self.model = Ridge()

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

        # Initialize the nearest neighbors model
        self.nn_model = NearestNeighbors(n_neighbors=self.k)

        # Fit the nearest neighbors model on the first k dimensions
        self.nn_model.fit(X[:, : self.nn_dimensions])

        # Construct new features
        X_new = self._construct_features(X)

        # Fit the linear regression model on the new features
        self.model.fit(X_new, y)

        return self

    def predict(self, X):
        # Construct new features for the test set
        X_new = self._construct_features(X)

        # Predict using the linear regression model
        return self.model.predict(X_new)

    def _construct_features(self, X):
        # Find the nearest neighbors
        distances, indices = self.nn_model.kneighbors(X[:, : self.nn_dimensions])

        # Calculate weights based on distances (inverse of distances)
        weights = 1 / (distances + 1e-5)  # Add a small value to avoid division by zero
        weights /= weights.sum(axis=1)[:, np.newaxis]  # Normalize weights

        if self.use_average:
            # Weighted average the features of the nearest neighbors
            X_new = np.sum(self.X_train[indices] * weights[:, :, np.newaxis], axis=1)
        else:
            # Concatenate the features of the nearest neighbors and apply weights
            weighted_features = self.X_train[indices] * weights[:, :, np.newaxis]
            X_new = weighted_features.reshape(X.shape[0], -1)

        return X_new


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    model = InContextLearnerRegressor(k=3, nn_dimensions=2, use_average=True)
    print("Cross-validated MSE: ", cross_val_score(model, X, y, cv=5).mean())
    model = KNeighborsRegressor()
    print("Cross-validated MSE: ", cross_val_score(model, X, y, cv=5).mean())
