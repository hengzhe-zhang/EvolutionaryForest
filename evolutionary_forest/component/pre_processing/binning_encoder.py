import os

import numpy as np
from sklearn import datasets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class KMeansBinningEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders_ = []  # To store one-hot encoders for each column
        self.k_values_ = []  # To store optimal k for each column
        self.cluster_centers_ = []  # To store cluster centers for each column

    def _get_optimal_k(self, data):
        max_k = min(
            10, len(data) - 1
        )  # Max k value can be less if the data has too few unique values
        silhouettes = []

        for k in range(2, max_k + 1):  # At least 2 clusters
            kmeans = MiniBatchKMeans(n_clusters=k, n_init="auto").fit(
                data.reshape(-1, 1)
            )
            silhouette_avg = silhouette_score(data.reshape(-1, 1), kmeans.labels_)
            silhouettes.append(silhouette_avg)

        optimal_k = (
            silhouettes.index(max(silhouettes)) + 2
        )  # +2 because our range starts from 2
        return optimal_k

    def fit(self, X, y=None):
        for col in range(X.shape[1]):
            data = X[:, col].reshape(-1, 1)

            # Get the optimal number of clusters for the column
            k = self._get_optimal_k(data)
            self.k_values_.append(k)

            # Apply k-means clustering
            kmeans = MiniBatchKMeans(n_clusters=k, n_init="auto").fit(data)
            self.cluster_centers_.append(kmeans.cluster_centers_)

            # One-hot encode the labels
            encoder = OneHotEncoder(sparse=False, categories="auto")
            encoder.fit(kmeans.labels_.reshape(-1, 1))
            self.encoders_.append(encoder)

        return self

    def transform(self, X):
        transformed_data = []

        for col in range(X.shape[1]):
            data = X[:, col].reshape(-1, 1)

            # Reshape data for broadcasting
            data_reshaped = data.reshape(-1, 1, 1)
            centers = self.cluster_centers_[col].reshape(1, -1, 1)

            # Calculate Euclidean distance
            distance_matrix = np.linalg.norm(data_reshaped - centers, axis=2)
            labels = np.argmin(distance_matrix, axis=1)

            # One-hot encode the labels
            encoded_data = self.encoders_[col].transform(labels.reshape(-1, 1))
            transformed_data.append(encoded_data)

        # Concatenate one-hot encoded features with original features
        return np.hstack([X] + transformed_data)


if __name__ == "__main__":
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    lr = RidgeCV()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    r2_before_encoding = r2_score(y_test, y_pred)

    print(f"R² score before encoding: {r2_before_encoding:.4f}")
    transformer = KMeansBinningEncoder()
    X_train_encoded = transformer.fit_transform(X_train)
    X_test_encoded = transformer.transform(
        X_test
    )  # Note: use transform, not fit_transform here

    lr_encoded = RidgeCV()
    lr_encoded.fit(X_train_encoded, y_train)
    y_pred_encoded = lr_encoded.predict(X_test_encoded)
    r2_after_encoding = r2_score(y_test, y_pred_encoded)

    print(f"R² score after encoding: {r2_after_encoding:.4f}")
    print(f"Difference in R² scores: {r2_after_encoding - r2_before_encoding:.4f}")
