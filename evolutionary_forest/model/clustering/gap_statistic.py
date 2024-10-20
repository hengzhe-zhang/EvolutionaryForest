import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from functools import lru_cache

from evolutionary_forest.model.cosine_kmeans import CosineKMeans


def calculate_wk(X, labels, cluster_centers):
    """
    Calculate the within-cluster dispersion using cosine distance for given labels and cluster centers.
    """
    n_clusters = cluster_centers.shape[0]
    Wk = 0
    for k in range(n_clusters):
        cluster_points = X[labels == k]

        # Skip empty clusters
        if len(cluster_points) == 0:
            continue

        # Calculate pairwise cosine distance between cluster points and cluster center
        distances = pairwise_distances(
            cluster_points, cluster_centers[k].reshape(1, -1), metric="cosine"
        )
        Wk += np.sum(distances**2)
    return Wk


@lru_cache(maxsize=None)
def cached_reference_wk(B, k_values, n_samples, n_features):
    """
    Cache the Wk for B reference datasets and return an array of size (B, len(k_values))
    with the Wk values for each k in k_values.

    Parameters:
    - B: Number of reference datasets to generate
    - k_values: List of cluster numbers to compute Wk for
    - n_samples: Number of samples in the dataset
    - n_features: Number of features in the dataset

    Returns:
    - Wk_ref: Cached Wk values for reference datasets (B x len(k_values))
    """
    Wk_ref = np.zeros((B, len(k_values)))

    # Generate B random reference datasets and calculate Wk for each
    for b in range(B):
        X_ref = np.random.rand(
            n_samples, n_features
        )  # Generate a uniform reference dataset
        for i, k in enumerate(k_values):
            kmeans = CosineKMeans(n_clusters=k).fit(X_ref)
            Wk_ref[b, i] = calculate_wk(X_ref, kmeans.labels_, kmeans.cluster_centers_)

    return Wk_ref


def gap_statistic(X, k_values, B=10):
    """
    Calculate the Gap Statistic for k in k_values, using caching of reference Wk based on B and the given k_values.

    Parameters:
    - X: Input data (n_samples, n_features)
    - k_values: List of cluster numbers to test
    - B: Number of bootstrapped reference datasets

    Returns:
    - optimal_k: The optimal number of clusters
    """
    n_samples, n_features = X.shape
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: Calculate Wk for real data for different values of k
    Wk_real = []
    for k in k_values:
        kmeans = CosineKMeans(n_clusters=k).fit(X_scaled)
        Wk_real.append(calculate_wk(X_scaled, kmeans.labels_, kmeans.cluster_centers_))

    # Step 2: Retrieve or calculate the cached Wk values for the reference datasets
    Wk_ref = cached_reference_wk(B, tuple(k_values), n_samples, n_features)

    # Step 3: Calculate the Gap statistic
    Wk_ref_log = np.log(np.mean(Wk_ref, axis=0))
    Wk_real_log = np.log(Wk_real)
    gap = Wk_ref_log - Wk_real_log

    # Step 4: Calculate the optimal number of clusters based on the Gap Statistic
    sk = np.std(np.log(Wk_ref), axis=0) * np.sqrt(1 + 1 / B)
    for i in range(len(k_values) - 1):
        if gap[i] >= gap[i + 1] - sk[i + 1]:
            return k_values[i]
    return k_values[-1]


if __name__ == "__main__":
    # Example usage with your dataset
    iris = load_iris()
    X = iris.data  # Using the feature data from the Iris dataset

    k_values = [2, 3, 5, 7]
    optimal_k = gap_statistic(X, k_values, B=10)
    print(f"Optimal number of clusters: {optimal_k}")
