import numpy as np
from scipy.spatial import KDTree, cKDTree


def retrieve_nearest_y_skip_self(train_tensors, train_targets, k=1):
    """
    Retrieve the k-nearest `y` from training data for each training sample,
    skipping the sample itself, using a KD-Tree.

    :param train_tensors: Tensor of shape (n_train_samples, input_dim).
    :param train_targets: List of `y` targets corresponding to train_tensors.
    :param k: Number of nearest neighbors to retrieve (excluding the sample itself).
    :return: List of concatenated nearest `y` values from train_targets for each training sample.
    """
    # Convert train_tensors to numpy array if it's not already
    train_tensors_np = train_tensors

    # Build a KD-Tree with the training data
    kdtree = KDTree(train_tensors_np)

    # Query the KD-Tree for the k+1 nearest neighbors of each point (k+1 to skip the point itself)
    distances, indices = kdtree.query(train_tensors_np, k=k + 1)

    # The nearest neighbors start from index 1 to k+1 in the result (skip index 0)
    nearest_indices = indices[:, 1 : k + 1]

    # Retrieve the corresponding `y` values from the targets and concatenate them
    nearest_y = [
        np.concatenate([train_targets[idx] for idx in neighbors])
        for neighbors in nearest_indices
    ]

    return nearest_y, kdtree  # Optionally return the KD-Tree if needed


def retrieve_nearest_y(kd_tree: cKDTree, train_targets, val_tensors, k=1):
    """
    Retrieve the k-nearest `y` from training data for each validation sample.

    :param kd_tree: Pre-built cKDTree from training data.
    :param train_targets: List of `y` targets corresponding to train_tensors.
    :param val_tensors: Tensor of shape (n_val_samples, input_dim).
    :param k: Number of nearest neighbors to retrieve.
    :return: List of concatenated nearest `y` values from train_targets for each validation sample.
    """
    # Convert val_tensors to numpy array if it's not already
    val_tensors_np = val_tensors

    # Query the KD-Tree for the k nearest neighbors of each validation point
    distances, nearest_indices = kd_tree.query(val_tensors_np, k=k)

    # Ensure nearest_indices is 2D for consistent processing (in case k=1)
    if k == 1:
        nearest_indices = nearest_indices[:, np.newaxis]

    # Retrieve the corresponding `y` values from the targets and concatenate them
    nearest_y = [
        np.concatenate([train_targets[idx] for idx in neighbors])
        for neighbors in nearest_indices
    ]

    return nearest_y
