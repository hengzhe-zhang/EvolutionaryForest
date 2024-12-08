import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils.validation import check_array, check_is_fitted

from evolutionary_forest.model.weight_solver import (
    solve_transformation_matrix,
    compute_lambda_matrix,
)


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
    def __init__(
        self,
        n_neighbors=5,
        distance="Euclidean",
        alpha=1.0,
        mode="full",
        random_seed=0,
        **params
    ):
        """
        mode: str, either "full" or "split"
            - "full": Use all features for KNN, then use KNN predictions concatenated with all features for Ridge.
            - "split": Use half of the features for KNN and the remaining half concatenated with KNN predictions for Ridge.
        """
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.alpha = alpha
        self.mode = mode
        self.random_seed = random_seed
        self.knn_with_gp = WeightedKNNWithGP(
            n_neighbors=n_neighbors,
            distance=distance,
            random_seed=random_seed,
            **params
        )
        self.ridge = Ridge(alpha=self.alpha)
        self.random_state = np.random.RandomState(self.random_seed)

    def fit(self, GP_X, y):
        self.coef_ = np.ones(GP_X.shape[1])

        if self.mode == "split" and GP_X.shape[1] > 1:
            # Split the features into two halves if there are more than one feature
            half_features = GP_X.shape[1] // 2
            GP_X_knn = GP_X[:, :half_features]  # First half of features for KNN
            GP_X_ridge = GP_X[:, half_features:]  # Second half of features for Ridge

            # Fit the KNN on the first half of features
            self.knn_with_gp.fit(GP_X_knn, y)

            # Predict on the full data with the KNN model trained on half the features
            knn_predictions = self.knn_with_gp.predict(GP_X_knn).reshape(-1, 1)

            # Concatenate the second half of features with KNN predictions
            combined_features = np.hstack((GP_X_ridge, knn_predictions))

            # Fit the Ridge model on the combined features
            self.ridge.fit(combined_features, y)

        else:
            # Fall back to full mode if only one feature is available
            self.knn_with_gp.fit(GP_X, y)
            knn_predictions = self.knn_with_gp.predict(GP_X).reshape(-1, 1)
            combined_features = np.hstack((GP_X, knn_predictions))
            self.ridge.fit(combined_features, y)

        return self

    def predict(self, x_test):
        if self.mode == "split" and x_test.shape[1] > 1:
            # Use only the first half of features for KNN predictions in split mode
            half_features = x_test.shape[1] // 2
            x_test_knn = x_test[:, :half_features]
            x_test_ridge = x_test[:, half_features:]

            # Predict using KNN with the first half of features
            knn_predictions = self.knn_with_gp.predict(x_test_knn).reshape(-1, 1)

            # Concatenate the second half of features with KNN predictions
            combined_features = np.hstack((x_test_ridge, knn_predictions))

        else:
            # Fall back to full mode if only one feature is available
            knn_predictions = self.knn_with_gp.predict(x_test).reshape(-1, 1)
            combined_features = np.hstack((x_test, knn_predictions))

        # Use the Ridge model to predict based on the combined features
        final_predictions = self.ridge.predict(combined_features)

        return final_predictions


class WeightedKNNWithGP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_neighbors=5,
        distance="Euclidean",
        random_seed=0,
        n_groups=1,
        reduced_dimension=None,
        weighted_instance=False,
        **params
    ):
        self.n_neighbors = n_neighbors
        self.distance = distance
        self.random_seed = random_seed
        self.n_groups = n_groups  # Number of groups for the ensemble
        self.random_state = np.random.RandomState(self.random_seed)
        self.reduced_dimension = reduced_dimension

        # Initialize KNN regressor based on distance type
        if distance == "Softmax":
            self.knn = SoftmaxWeightedKNNRegressor(
                n_neighbors=self.n_neighbors, weights="distance"
            )
        else:
            self.knn = KNeighborsRegressor(
                n_neighbors=self.n_neighbors, weights="distance"
            )

        self.weights = []  # List to store transformation matrices for each group
        self.weighted_instance = weighted_instance

    def fit(self, GP_X, y):
        # Determine if we need to subsample
        if len(y) > 100:
            subsample_indices = self.random_state.choice(len(y), 100, replace=False)
            GP_X_subsample = GP_X[subsample_indices]
            y_subsample = y[subsample_indices]
        else:
            GP_X_subsample = GP_X
            y_subsample = y

        # Initialize the coefficient vector
        self.coef_ = np.ones(GP_X_subsample.shape[1])
        # Divide GP_X into `n_groups` subsets and calculate weights for each
        group_size = GP_X_subsample.shape[0] // self.n_groups
        self.weights = []

        for i in range(self.n_groups):
            # Define the group indices
            start_idx = i * group_size
            end_idx = (
                (i + 1) * group_size
                if i < self.n_groups - 1
                else GP_X_subsample.shape[0]
            )
            GP_X_group = GP_X_subsample[start_idx:end_idx]
            y_group = y_subsample[start_idx:end_idx]

            # Compute the distance matrix D for this subset
            D_group = pairwise_distances(y_group.reshape(-1, 1), metric="euclidean")

            # Compute the transformation matrix for this group
            # reduced_dimension = GP_X_group.shape[1]
            if self.reduced_dimension == None:
                reduced_dimension = GP_X_group.shape[1]
            else:
                reduced_dimension = self.reduced_dimension

            if self.weighted_instance:
                weights = compute_lambda_matrix(y_group)
            else:
                weights = None

            weight = solve_transformation_matrix(
                GP_X_group,
                D_group,
                weights=weights,
                p=reduced_dimension,
            )
            self.print_mse(D_group, GP_X_group, weight)
            self.weights.append(weight)

        # Transform the entire training data using the computed weights and concatenate them
        transformed_data_list = [GP_X @ weight.T for weight in self.weights]
        training_data = np.concatenate(transformed_data_list, axis=1)
        self.training_data = training_data

        # Fit the KNN model on the concatenated weighted transformed space
        self.knn.fit(training_data, y)

        return self

    def print_mse(self, D_group, GP_X_group, weight):
        print(
            "Original R2",
            r2_score(
                pairwise_distances(GP_X_group, metric="euclidean").flatten(),
                D_group.flatten(),
            ),
            "R2",
            r2_score(
                pairwise_distances(GP_X_group @ weight.T, metric="euclidean").flatten(),
                D_group.flatten(),
            ),
        )

    def predict(self, x_test):
        # Transform test data using each weight matrix and concatenate the results
        test_data_list = [x_test @ weight.T for weight in self.weights]
        test_data = np.concatenate(test_data_list, axis=1)

        # Predict using the KNN model
        prediction = self.knn.predict(test_data)
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
