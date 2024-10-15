import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances, r2_score
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
    def __init__(self, n_neighbors=5, distance="euclidean", **params):
        self.n_neighbors = n_neighbors

        # Initialize KNN regressor based on knn_type
        if distance == "Softmax":
            self.knn = SoftmaxWeightedKNNRegressor(
                n_neighbors=self.n_neighbors, weights="distance", metric="precomputed"
            )
        else:
            self.knn = KNeighborsRegressor(
                n_neighbors=self.n_neighbors, weights="distance", metric="precomputed"
            )

        self.W = None  # Transformation vector

    def fit(self, GP_X, y):
        self.coef_ = np.ones(GP_X.shape[1])

        # Compute the original distance matrix D using labels
        D = pairwise_distances(y.reshape(-1, 1), metric="euclidean")

        # Compute the transformed distance matrix D' using GP_X features
        D_prime = pairwise_distances(GP_X, metric="euclidean")
        self.training_data = GP_X

        # Calculate the transformation matrix W
        D_prime_D_prime_T = D_prime @ D_prime.T
        if (
            np.linalg.matrix_rank(D_prime_D_prime_T) == D_prime_D_prime_T.shape[0]
        ):  # Check invertibility
            self.W = D @ D_prime.T @ np.linalg.inv(D_prime_D_prime_T)
        else:
            self.W = D @ D_prime.T @ np.linalg.pinv(D_prime_D_prime_T)

        # Transform GP_X using W to get the weighted distance
        weighted_GP_X = self.W @ D_prime  # Corrected for dimensional consistency
        weighted_GP_X[weighted_GP_X < 0] = 0  # Ensure non-negativity

        # Fit the KNN model on the weighted transformed space
        self.knn.fit(weighted_GP_X.T, y)

        return self

    def predict(self, x_test):
        # Transform GP_X using W
        D_prime = pairwise_distances(self.training_data, x_test, metric="euclidean")
        weighted_GP_X = self.W @ D_prime
        weighted_GP_X[weighted_GP_X < 0] = 0  # Ensure non-negativity

        # Predict using the KNN model
        prediction = self.knn.predict(weighted_GP_X.T)
        # np.any(np.isnan(prediction))
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return prediction


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
    model.fit(GP_X_train, y_train)

    # Make predictions
    y_pred = model.predict(GP_X_test)

    # Calculate and print the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print("Training R2 Score: ", r2_score(y_train, model.predict(GP_X_train)))
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
