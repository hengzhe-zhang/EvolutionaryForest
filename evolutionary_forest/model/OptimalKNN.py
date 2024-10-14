import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_array, check_is_fitted


class SoftmaxWeightedKNNRegressor(KNeighborsRegressor):
    def predict(self, X):
        """
        Predict using softmax-weighted KNN.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Test samples.

        Returns:
        y_pred : array, shape (n_samples,)
            Target values for each sample.
        """
        # Ensure the model is already fitted
        check_is_fitted(self, "_fit_X")
        X = check_array(X, accept_sparse="csr")

        # Find the nearest neighbors
        distances, neighbors = self.kneighbors(X)

        # Avoid division by zero in distance (add a small epsilon)
        distances = np.where(distances == 0, 1e-10, distances)

        # Compute softmax weights based on inverse distances
        softmax_weights = np.exp(-distances) / np.sum(
            np.exp(-distances), axis=1, keepdims=True
        )

        # Weighted predictions
        y_pred = np.sum(softmax_weights * self._y[neighbors], axis=1)

        return y_pred


class WeightedKNNWithGP(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        # self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance")
        # self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.knn = SoftmaxWeightedKNNRegressor(n_neighbors=self.n_neighbors)
        self.W = None  # Transformation matrix

    def fit(self, X, GP_X, y):
        """
        Fit the weighted KNN using both original features and GP-transformed features.

        Parameters:
        X    : Original feature matrix (n_samples, n_features)
        GP_X : GP-transformed feature matrix (n_samples, k)
        y    : Target variable (n_samples,)
        """
        # Compute the original distance matrix D using labels
        D = pairwise_distances(y.reshape(-1, 1), metric="euclidean")

        # Compute the transformed distance matrix D' using GP_X features
        D_prime = pairwise_distances(GP_X, metric="euclidean")

        # Calculate the transformation matrix W
        D_prime_D_prime_T = D_prime @ D_prime.T
        if (
            np.linalg.matrix_rank(D_prime_D_prime_T) == D_prime_D_prime_T.shape[0]
        ):  # Check invertibility
            self.W = D @ D_prime.T @ np.linalg.inv(D_prime_D_prime_T)
        else:
            self.W = D @ D_prime.T @ np.linalg.pinv(D_prime_D_prime_T)

        # Ensure W has correct dimensions for GP_X transformation
        if self.W.shape[1] != GP_X.shape[1]:
            # Take only the first `k` columns if W is oversized
            self.W = self.W[:, : GP_X.shape[1]]

        # Transform GP_X using W to get the weighted distance
        weighted_GP_X = GP_X @ self.W.T  # Corrected for dimensional consistency

        # Fit the KNN model on the weighted transformed space
        self.knn.fit(weighted_GP_X, y)

        return self

    def predict(self, X, GP_X):
        """
        Predict using the weighted KNN model.

        Parameters:
        X    : Original feature matrix (n_samples, n_features)
        GP_X : GP-transformed feature matrix (n_samples, k)

        Returns:
        Predicted values (n_samples,)
        """
        # Transform GP_X using W
        weighted_GP_X = GP_X @ self.W.T

        # Predict using the KNN model
        return self.knn.predict(weighted_GP_X)


if __name__ == "__main__":
    # Generating a random dataset to demonstrate the usage of the WeightedKNNWithGP model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Generate random data for original features X and GP-transformed features GP_X
    np.random.seed(0)
    n_samples = 100
    n_features = 4
    k = 20  # Number of GP-transformed features

    # Random original features
    # X, y = make_regression(
    #     n_samples=n_samples, n_features=n_features, noise=0.1, random_state=0
    # )
    X, y = load_diabetes(return_X_y=True)

    # Random GP-transformed features (simulating transformations from GP trees)
    GP_X = X @ np.random.rand(X.shape[1], k)

    # Split the data into training and testing sets
    X_train, X_test, GP_X_train, GP_X_test, y_train, y_test = train_test_split(
        X, GP_X, y, test_size=0.2, random_state=0
    )

    # Instantiate and train the model
    model = WeightedKNNWithGP(n_neighbors=5)
    model.fit(X_train, GP_X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test, GP_X_test)

    # Calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("Training R2 Score: ", r2_score(y_train, model.predict(X_train, GP_X_train)))
    print("R2 Score: ", r2_score(y_test, y_pred))

    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Training R2 Score: ", r2_score(y_train, model.predict(X_train)))
    print("KNN R2 Score: ", r2_score(y_test, y_pred))

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Training R2 Score: ", r2_score(y_train, model.predict(X_train)))
    print("Linear Regression R2 Score: ", r2_score(y_test, y_pred))
