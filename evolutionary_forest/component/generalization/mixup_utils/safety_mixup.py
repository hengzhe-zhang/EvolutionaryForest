import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

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


def safe_mixup(X, y, mixup_bandwidth, alpha_beta=None, random_seed=0):
    np.random.seed(random_seed)
    distance_matrix = rbf_kernel(y.reshape(-1, 1), gamma=mixup_bandwidth)

    ratio = np.random.beta(alpha_beta, alpha_beta, len(X))

    # Indices for data generation
    indices_a = np.random.randint(0, len(X), len(X))
    indices_b = sample_according_to_distance(distance_matrix, indices_a)

    # Generate initial synthetic data
    data, label = create_synthetic_data(X, y, indices_a, indices_b, ratio)

    # Nearest Neighbor to check conformation
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    _, indices_nn = nbrs.kneighbors(data)  # Find nearest neighbors for synthetic data
    y_nn = y[indices_nn].flatten()

    # Conformity check and regeneration
    max_retries = 10  # Limit to prevent infinite loops
    retry_counter = 0
    for idx in range(len(data)):
        y_i, y_j = y[indices_a[idx]], y[indices_b[idx]]
        retries = 0
        while (
            y_nn[idx] > max(y_i, y_j) or y_nn[idx] < min(y_i, y_j)
        ) and retries < max_retries:
            # Regenerate this particular data point
            ratio[idx] = np.random.beta(alpha_beta, alpha_beta)
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
            _, index_nn = nbrs.kneighbors([data[idx]])
            y_nn[idx] = y[index_nn].flatten()[0]
            retries += 1
            retry_counter += 1
    # print(f"Total retries: {retry_counter}")

    return data, label, ((indices_a, ratio), (indices_b, 1 - ratio))
