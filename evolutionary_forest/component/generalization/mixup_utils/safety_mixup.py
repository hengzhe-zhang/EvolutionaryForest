import numpy as np
from sklearn.base import RegressorMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, IsolationForest, ExtraTreesRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import (
    KNeighborsRegressor,
    RadiusNeighborsRegressor,
)

from evolutionary_forest.component.generalization.mixup_utils.automatic_mixup import (
    compute_mixup_ratio,
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


def safe_mixup_with_minifold_intrusion_detection(
    X,
    y,
    kernel_space,
    distance_matrix,
    mixup_bandwidth,
    alpha_beta=None,
    mode="",
    skip_self=False,
):
    confidence_interval = 1
    if len(mode.split(",")) == 3:
        mixup_flag, retry_flag, confidence_interval = mode.split(",")
        confidence_interval = float(confidence_interval)
    else:
        mixup_flag, retry_flag = mode.split(",")

    # Step 2: Perform clustering
    cluster_labels = None
    if mixup_flag == "Clustering":
        kmeans = KMeans(n_clusters=mixup_bandwidth)
        cluster_labels = kmeans.fit_predict(kernel_space)
    elif mixup_flag == "Clustering+RBF":
        n_clusters = int(np.sqrt(len(X)))  # Determine number of clusters
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(kernel_space)

    # Step 3: Generate initial synthetic data
    # Indices for data generation
    indices_a = np.arange(0, len(X))
    if mixup_flag == "Clustering":
        assert cluster_labels is not None
        indices_b = sample_indices_from_same_cluster(cluster_labels)
    elif mixup_flag == "Clustering+RBF":
        indices_b = sample_indices_within_cluster(
            cluster_labels, kernel_space, mixup_bandwidth
        )
    else:
        assert mixup_flag == "RBF"
        indices_b = sample_according_to_distance(
            distance_matrix, indices_a, skip_self=skip_self
        )

    ratio = np.random.beta(alpha_beta, alpha_beta, len(X))
    if alpha_beta == "Adaptive":
        ratio = compute_mixup_ratio(distance_matrix, indices_a, indices_b)
    ratio = np.where(ratio < 1 - ratio, 1 - ratio, ratio)

    # Generate initial synthetic data
    data, label = create_synthetic_data(X, y, indices_a, indices_b, ratio)

    # Step 4: Nearest Neighbor to check conformation
    model = None
    if retry_flag == "KNN-3":
        model = KNeighborsRegressor(n_neighbors=3, weights="distance")
        model.fit(X, y)
        y_nn = model.predict(data)
    elif retry_flag == "KNN-1":
        model = KNeighborsRegressor(n_neighbors=1)
        model.fit(X, y)
        y_nn = model.predict(data)
    elif retry_flag == "KNN-3+":
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
    elif retry_flag == "ET":
        model = ExtraTreesRegressor(n_estimators=100)
        model.fit(X, y)
        y_nn = model.predict(data)
    elif retry_flag == "IsolationForest":
        model = IsolationForest()
        model.fit(X)
        y_nn = model.predict(data)

    # Step 5: Conformity check and regeneration
    max_retries = 100
    retry_counter = 0
    for idx in range(len(data)):
        y_i, y_j = y[indices_a[idx]], y[indices_b[idx]]
        retries = 0
        increased_alpha_beta = alpha_beta
        while (
            (
                isinstance(model, RegressorMixin)
                and condition_of_prediction_based_check(
                    idx,
                    y_i,
                    y_j,
                    y_nn,
                    ratio[idx],
                    confidence_interval=confidence_interval,
                )
            )
            or (
                isinstance(model, IsolationForest)
                and condition_of_outlier_based_check(idx, y_i, y_j, y_nn)
            )
        ) and retries < max_retries:
            # Regenerate this particular data point
            if retries > 0 and retries % 10 == 0:
                increased_alpha_beta = increased_alpha_beta * 10
                # print("Increase", increased_alpha_beta)
            indices_b[idx] = sample_according_to_distance(
                distance_matrix, indices_a[idx : idx + 1], skip_self=skip_self
            )[
                0
            ]  # Sample new b
            ratio[idx] = np.random.beta(increased_alpha_beta, alpha_beta)
            if alpha_beta == "Adaptive":
                ratio = compute_mixup_ratio(
                    distance_matrix, indices_a[idx], indices_b[idx]
                )
            ratio[idx] = np.where(
                ratio[idx] < 1 - ratio[idx], 1 - ratio[idx], ratio[idx]
            )
            y_i, y_j = y[indices_a[idx]], y[indices_b[idx]]
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
    #     if retries >= max_retries:
    #         print("Max retries reached for a data point!")
    # print(f"Total retries: {retry_counter}")

    return data, label, ((indices_a, ratio), (indices_b, 1 - ratio))


def condition_of_outlier_based_check(idx, y_i, y_j, y_nn):
    return y_nn[idx] == -1


def condition_of_prediction_based_check(
    idx, y_i, y_j, y_nn, ratio, confidence_interval=0.1
):
    ratio_lb = max(ratio - confidence_interval, 0)
    ratio_ub = min(ratio + confidence_interval, 1)
    y_lb = y_i * ratio_lb + y_j * (1 - ratio_lb)
    y_ub = y_i * ratio_ub + y_j * (1 - ratio_ub)
    # calculate the lower and upper bound of the prediction
    if y_lb > y_ub:
        y_lb, y_ub = y_ub, y_lb

    if y_lb > y_nn[idx] or y_nn[idx] > y_ub:
        return True
    return False
