import numpy as np
from sklearn.base import RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import (
    NearestNeighbors,
    KNeighborsRegressor,
    RadiusNeighborsRegressor,
)

from evolutionary_forest.utility.sampling_utils import sample_according_to_distance


def create_synthetic_data(X, y, indices_a, indices_b, ratios):
    data = X[indices_a] * ratios.reshape(-1, 1) + X[indices_b] * (
        1 - ratios.reshape(-1, 1)
    )
    labels = y[indices_a] * ratios + y[indices_b] * (1 - ratios)
    return data, labels


def create_point(x_a, x_b, y_a, y_b, ratio):
    data_point = x_a * ratio + x_b * (1 - ratio)
    label_point = y_a * ratio + y_b * (1 - ratio)
    return data_point, label_point


def sample_indices_from_same_cluster(cluster_labels):
    unique_clusters = np.unique(cluster_labels)
    cluster_to_indices = {
        cluster: np.where(cluster_labels == cluster)[0] for cluster in unique_clusters
    }

    indices_b = np.zeros(len(cluster_labels), dtype=int)

    for idx, cluster in enumerate(cluster_labels):
        possible_indices = cluster_to_indices[cluster]
        possible_indices = possible_indices[
            possible_indices != idx
        ]  # Exclude the current index
        if len(possible_indices) == 0:  # If there are no other indices in the cluster
            indices_b[
                idx
            ] = idx  # Fallback to itself, though this case should be rare with enough data
        else:
            indices_b[idx] = np.random.choice(possible_indices)

    return indices_b


def sample_indices_within_cluster(cluster_labels, data, mixup_bandwidth):
    n_clusters = len(np.unique(cluster_labels))
    indices_b = np.arange(len(data))

    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) < 2:
            continue
        distance_matrix = rbf_kernel(data[cluster_indices], gamma=mixup_bandwidth)
        indices = np.arange(len(cluster_indices))
        probabilities = distance_matrix / distance_matrix.sum(axis=1, keepdims=True)
        indices_b[cluster_indices] = np.array(
            [np.random.choice(indices, p=probabilities[i]) for i in range(len(indices))]
        )

    return indices_b


def safe_mixup(X, y, mixup_bandwidth, alpha_beta=None, mode="Clustering"):
    # Step 1: Compute distance matrix
    distance_matrix = rbf_kernel(y.reshape(-1, 1), gamma=mixup_bandwidth)
    mixup_flag, retry_flag = mode.split(",")

    # Step 2: Perform clustering
    cluster_labels = None
    if mixup_flag == "Clustering":
        kmeans = KMeans(n_clusters=mixup_bandwidth)
        cluster_labels = kmeans.fit_predict(y.reshape(-1, 1))
    elif mixup_flag == "Clustering+RBF":
        n_clusters = int(np.sqrt(len(X)))  # Determine number of clusters
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(y.reshape(-1, 1))

    # Step 3: Generate initial synthetic data
    ratio = np.random.beta(alpha_beta, alpha_beta, len(X))
    ratio = np.where(ratio < 1 - ratio, 1 - ratio, ratio)

    # Indices for data generation
    indices_a = np.arange(0, len(X))
    if mixup_flag == "Clustering":
        assert cluster_labels is not None
        indices_b = sample_indices_from_same_cluster(cluster_labels)
    elif mixup_flag == "Clustering+RBF":
        indices_b = sample_indices_within_cluster(
            cluster_labels, y.reshape(-1, 1), mixup_bandwidth
        )
    else:
        assert mixup_flag == "RBF"
        indices_b = sample_according_to_distance(distance_matrix, indices_a)

    # Generate initial synthetic data
    data, label = create_synthetic_data(X, y, indices_a, indices_b, ratio)

    # Step 4: Nearest Neighbor to check conformation
    model = None
    if retry_flag == "KNN-3":
        model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X, y)
        y_nn = model.predict(data)
    elif retry_flag == "KNN-5":
        model = KNeighborsRegressor(n_neighbors=3)
        model.fit(X, y)
        y_nn = model.predict(data)
    elif retry_flag == "RNN":
        model = RadiusNeighborsRegressor()
        model.fit(X, y)
        y_nn = model.predict(data)
    elif retry_flag == "RF":
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        y_nn = model.predict(data)
    elif retry_flag == "IsolationForest":
        model = IsolationForest()
        model.fit(X)
        y_nn = model.predict(data)

    # Step 5: Conformity check and regeneration
    max_retries = 10
    retry_counter = 0
    for idx in range(len(data)):
        y_i, y_j = y[indices_a[idx]], y[indices_b[idx]]
        retries = 0
        while (
            (
                isinstance(model, RegressorMixin)
                and condition_of_prediction_based_check(idx, y_i, y_j, y_nn)
            )
            or (
                isinstance(model, IsolationForest)
                and condition_of_outlier_based_check(idx, y_i, y_j, y_nn)
            )
        ) and retries < max_retries:
            # Regenerate this particular data point
            ratio[idx] = np.random.beta(alpha_beta, alpha_beta)
            ratio = np.where(ratio < 1 - ratio, 1 - ratio, ratio)
            indices_b[idx] = sample_according_to_distance(
                distance_matrix, indices_a[idx : idx + 1]
            )[
                0
            ]  # Sample new b
            data[idx], label[idx] = create_point(
                X[indices_a[idx]],
                X[indices_b[idx]],
                y[indices_a[idx]],
                y[indices_b[idx]],
                ratio[idx],
            )
            # Check its nearest neighbor again
            y_nn[idx] = model.predict([data[idx]])[0]
            retries += 1
            retry_counter += 1
    # print(f"Total retries: {retry_counter}")

    return data, label, ((indices_a, ratio), (indices_b, 1 - ratio))


def condition_of_outlier_based_check(idx, y_i, y_j, y_nn):
    return y_nn[idx] == -1


def condition_of_prediction_based_check(idx, y_i, y_j, y_nn):
    return y_nn[idx] > max(y_i, y_j) or y_nn[idx] < min(y_i, y_j)
