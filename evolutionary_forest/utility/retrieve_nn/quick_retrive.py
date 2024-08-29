from scipy.spatial import KDTree, cKDTree


def retrieve_nearest_y_skip_self(train_tensors, train_targets):
    """
    Retrieve the 1-nearest `y` from training data for each training sample,
    skipping the sample itself, using a KD-Tree.

    :param train_tensors: Tensor of shape (n_train_samples, input_dim).
    :param train_targets: List of `y` targets corresponding to train_tensors.
    :return: List of nearest `y` from train_targets for each training sample.
    """
    # Convert train_tensors to numpy array if it's not already
    train_tensors_np = train_tensors

    # Build a KD-Tree with the training data
    kdtree = KDTree(train_tensors_np)

    # Query the KD-Tree for the nearest neighbor of each point
    # k=2 because the nearest neighbor will be the point itself, so we need the second nearest
    distances, indices = kdtree.query(train_tensors_np, k=2)

    # The second nearest neighbor (index 1 in the result) is the nearest that is not the point itself
    nearest_indices = indices[:, 1]

    # Retrieve the corresponding `y` values from the targets
    nearest_y = [train_targets[idx] for idx in nearest_indices]

    return nearest_y, kdtree  # Optionally return the KD-Tree if needed


def retrieve_nearest_y(kd_tree: cKDTree, train_targets, val_tensors):
    """
    Retrieve the 1-nearest `y` from training data for each validation sample.

    :param train_tensors: Tensor of shape (n_train_samples, input_dim).
    :param train_targets: List of `y` targets corresponding to train_tensors.
    :param val_tensors: Tensor of shape (n_val_samples, input_dim).
    :return: List of nearest `y` from train_targets for each validation sample.
    """
    # Convert val_tensors to numpy array if it's not already
    val_tensors_np = val_tensors

    # Query the KD-Tree for the nearest neighbor of each validation point
    distances, nearest_indices = kd_tree.query(val_tensors_np, k=1)

    # Retrieve the corresponding `y` values from the targets
    nearest_y = [train_targets[idx] for idx in nearest_indices]

    return nearest_y
