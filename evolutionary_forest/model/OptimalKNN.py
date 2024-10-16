import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_array, check_is_fitted

from evolutionary_forest.model.weight_solver import solve_transformation_matrix


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


class WeightedKNNWithGPRidge(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, distance="Euclidean", alpha=1.0, **params):
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.alpha = alpha
        self.knn_with_gp = WeightedKNNWithGP(
            n_neighbors=n_neighbors, distance=distance, **params
        )
        self.ridge = Ridge(alpha=self.alpha)

    def fit(self, GP_X, y):
        self.coef_ = np.ones(GP_X.shape[1])

        # Fit the Weighted KNN with GP transformation
        self.knn_with_gp.fit(GP_X, y)

        # Predict on the training data using the fitted WeightedKNNWithGP
        knn_predictions = self.knn_with_gp.predict(GP_X).reshape(-1, 1)

        # Concatenate the original features (GP_X) and the KNN predictions
        combined_features = np.hstack((GP_X, knn_predictions))

        # Fit the Ridge regression using the concatenated features
        self.ridge.fit(combined_features, y)

        return self

    def predict(self, x_test):
        # Get predictions from the KNN model for x_test
        knn_predictions = self.knn_with_gp.predict(x_test).reshape(-1, 1)

        # Concatenate the original features (x_test) and the KNN predictions
        combined_features = np.hstack((x_test, knn_predictions))

        # Use the Ridge model to predict based on the combined features
        final_predictions = self.ridge.predict(combined_features)

        return final_predictions


class WeightedKNNWithGP(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, distance="Euclidean", **params):
        self.n_neighbors = n_neighbors
        self.distance = distance

        # Initialize KNN regressor based on knn_type
        if distance == "Softmax":
            self.knn = SoftmaxWeightedKNNRegressor(
                n_neighbors=self.n_neighbors, weights="distance"
            )
        else:
            self.knn = KNeighborsRegressor(
                n_neighbors=self.n_neighbors, weights="distance"
            )

        self.W = None  # Transformation vector

    def fit(self, GP_X, y):
        self.coef_ = np.ones(GP_X.shape[1])

        # Compute the original distance matrix D using labels
        D = pairwise_distances(y.reshape(-1, 1), metric="euclidean")

        # Compute the transformed distance matrix D' using GP_X features
        weight = solve_transformation_matrix(GP_X, D, p=GP_X.shape[1])
        training_data = GP_X @ weight

        self.weight = weight
        self.training_data = training_data

        # Fit the KNN model on the weighted transformed space
        self.knn.fit(training_data, y)

        return self

    def predict(self, x_test):
        # Transform GP_X using W
        test_data = x_test @ self.weight

        # Predict using the KNN model
        prediction = self.knn.predict(test_data)
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
