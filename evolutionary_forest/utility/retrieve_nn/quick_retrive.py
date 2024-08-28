import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform


def retrieve_nearest_y_skip_self(train_tensors, train_targets):
    """
    Retrieve the 1-nearest `y` from training data for each training sample,
    skipping the sample itself.

    :param train_tensors: Tensor of shape (n_train_samples, input_dim).
    :param train_targets: List of `y` targets corresponding to train_tensors.
    :return: List of nearest `y` from train_targets for each training sample.
    """
    # Convert tensors to numpy arrays for distance computation
    train_tensors_np = train_tensors

    # Compute pairwise distances between all training samples
    distances = pdist(train_tensors_np, metric="euclidean")

    # Convert to a square distance matrix
    distance_matrix = squareform(distances)

    # Set the diagonal to infinity to skip self-comparison
    np.fill_diagonal(distance_matrix, np.inf)

    # Find the index of the nearest training sample for each sample
    nearest_indices = np.argmin(distance_matrix, axis=1)

    nearest_y = [train_targets[idx] for idx in nearest_indices]

    return nearest_y


def retrieve_nearest_y(train_tensors, train_targets, val_tensors):
    """
    Retrieve the 1-nearest `y` from training data for each validation sample.

    :param train_tensors: Tensor of shape (n_train_samples, input_dim).
    :param train_targets: List of `y` targets corresponding to train_tensors.
    :param val_tensors: Tensor of shape (n_val_samples, input_dim).
    :return: List of nearest `y` from train_targets for each validation sample.
    """
    # Convert tensors to numpy arrays for distance computation
    train_tensors_np = train_tensors
    val_tensors_np = val_tensors

    # Compute pairwise distances between val_tensors and train_tensors
    distances = cdist(val_tensors_np, train_tensors_np, metric="euclidean")

    # Find the index of the nearest training sample for each validation sample
    nearest_indices = np.argmin(distances, axis=1)

    nearest_y = [train_targets[idx] for idx in nearest_indices]

    return nearest_y
