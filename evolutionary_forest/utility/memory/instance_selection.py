import numpy as np
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist


def informed_down_sampling(population_errors, n_samples):
    """
    Performs informed down-sampling to select diverse training cases.

    Parameters:
    - population_errors (array-like): Errors of individuals on training cases (matrix).
    - n_samples (int): Number of training cases to sample.

    Returns:
    - selected_cases (array-like): Indices of selected diverse training cases.
    """
    # Infer the number of training cases from population_errors
    n_cases = population_errors.shape[1]

    # Compute Euclidean distances between training cases based on errors
    distances = cdist(
        population_errors.T, population_errors.T, metric="euclidean"
    )  # Transpose for case-wise distance

    # Farthest-First Traversal to select diverse cases
    selected_cases = []
    selected_cases.append(np.random.choice(n_cases))  # Start with a random case
    while len(selected_cases) < n_samples:
        dist_to_selected = np.min(distances[selected_cases], axis=0)
        next_case = np.argmax(dist_to_selected)
        selected_cases.append(next_case)

    return np.array(selected_cases)


def semantic_instance_selection(loss_matrix, semantics_length):
    # Step 1: Transpose the loss matrix to get instance-wise losses
    instance_losses = loss_matrix.T  # Shape: (num_instances, num_individuals)

    norms = np.linalg.norm(instance_losses, axis=1)
    norms[norms == 0] = 1e-10  # Replace zero with a small epsilon
    normalized_instance_losses = (instance_losses.T / norms).T

    # Step 2: Cluster instances into `semantics_length` clusters
    kmeans = KMeans(n_clusters=semantics_length, random_state=0)
    cluster_labels = kmeans.fit_predict(normalized_instance_losses)

    # Step 3: Select the hardest instance from each cluster
    selected_indices = []
    for cluster in range(semantics_length):
        # Get the indices of instances in the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) == 0:
            continue

        # Compute the median loss for each instance in the cluster
        cluster_medians = np.median(loss_matrix[:, cluster_indices], axis=0)

        # Select the instance with the highest median loss
        hardest_instance_index = cluster_indices[np.argmax(cluster_medians)]
        selected_indices.append(hardest_instance_index)

    # Step 4: Return the selected indices
    return np.array(selected_indices)


if __name__ == "__main__":
    loss_matrix = np.random.rand(5, 15)
    semantics_length = 5
    # Perform instance selection
    selected_indices = semantic_instance_selection(loss_matrix, semantics_length)
    print("Selected Instance Indices:", selected_indices)
