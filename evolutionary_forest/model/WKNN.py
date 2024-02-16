import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


class GaussianKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, k):
        self.k = k

    def gaussian_kernel(self, distances, sigma=1):
        weights = np.exp(-0.5 * (distances / sigma) ** 2)
        return weights

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        knn = KNeighborsRegressor(n_neighbors=self.k)
        knn.fit(self.X_train, self.y_train)
        distances, indices = knn.kneighbors(X_test)

        y_pred = []
        for i in range(len(X_test)):
            weights = self.gaussian_kernel(distances[i])
            weighted_sum = np.sum(weights)
            weighted_avg = np.dot(weights, self.y_train[indices[i]]) / weighted_sum
            y_pred.append(weighted_avg)

        return np.array(y_pred)


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)

    # Define parameter combinations to compare
    test_sizes = [0.2, 0.5, 0.8]
    n_neighbors_values = [3, 5, 7, 10, 15, 20, 30]

    # Initialize lists to store R-squared scores for both models
    knn_r2_values = []
    weighted_knn_r2_values = []

    # Iterate over different test_size values
    for test_size in test_sizes:
        knn_r2_values_test_size = []
        weighted_knn_r2_values_test_size = []

        for n_neighbors in n_neighbors_values:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0
            )

            # Regular KNeighborsRegressor
            knn_regressor = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                # weights="distance"
            )
            knn_regressor.fit(X_train, y_train)
            knn_y_pred = knn_regressor.predict(X_test)
            knn_r2 = r2_score(y_test, knn_y_pred)
            knn_r2_values_test_size.append(knn_r2)

            # WeightedKNNRegressor
            weighted_knn_regressor = GaussianKNNRegressor(k=n_neighbors)
            weighted_knn_regressor.fit(X_train, y_train)
            weighted_knn_y_pred = weighted_knn_regressor.predict(X_test)
            weighted_knn_r2 = r2_score(y_test, weighted_knn_y_pred)
            weighted_knn_r2_values_test_size.append(weighted_knn_r2)

        # Append R-squared values for the current test_size
        knn_r2_values.append(knn_r2_values_test_size)
        weighted_knn_r2_values.append(weighted_knn_r2_values_test_size)

    # Plotting
    plt.figure(figsize=(10, 6))

    for i, test_size in enumerate(test_sizes):
        plt.plot(
            n_neighbors_values,
            knn_r2_values[i],
            marker="o",
            label=f"KNN, Test Size={test_size}",
        )
        plt.plot(
            n_neighbors_values,
            weighted_knn_r2_values[i],
            marker="o",
            label=f"Weighted KNN, Test Size={test_size}",
        )

    plt.title("Comparison of KNN and Weighted KNN Regressors")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("R-squared Score")
    plt.xticks(n_neighbors_values)
    plt.legend()
    plt.grid(True)
    plt.show()
