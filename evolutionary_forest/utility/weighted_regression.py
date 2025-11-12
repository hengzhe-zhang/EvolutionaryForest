import numpy as np
from sklearn.neighbors import NearestNeighbors


def vectorized_nonlinearity_local(X, y, k=None):
    """
    Vectorized version of nonlinearity_local_linear using matrix operations.
    Computes nonlinearity-based weights without Python loops.

    Parameters:
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
        k : number of nearest neighbors (default = d + 1)
    Returns:
        w : normalized nonlinearity weights (sum = 1)
    """
    n, d = X.shape
    if k is None:
        k = min(d + 1, n - 1)

    # Step 1: Find nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, neighbor_indices_all = nbrs.kneighbors(X)
    # Exclude self (first neighbor) and take k neighbors: shape (n, k)
    neighbor_indices = neighbor_indices_all[:, 1 : k + 1]

    # Step 2: Extract neighbor features and targets
    # X_neighbors: (n, k, d) - features of k neighbors for each of n points
    # y_neighbors: (n, k) - targets of k neighbors for each of n points
    X_neighbors = X[neighbor_indices]
    y_neighbors = y[neighbor_indices]

    # Step 3: Prepare data for linear regression (add intercept column)
    # LinearRegression includes intercept, so we add a column of ones
    intercept_column = np.ones((n, k, 1))
    X_neighbors_with_intercept = np.concatenate(
        [intercept_column, X_neighbors], axis=2
    )  # (n, k, d+1)

    # Step 4: Build normal equation matrices for all local regressions
    # For each point i, we solve: (X_i^T X_i) @ beta_i = X_i^T y_i
    # XTX_all: (n, d+1, d+1) - X^T X for each point
    # XTy_all: (n, d+1) - X^T y for each point
    XTX_all = X_neighbors_with_intercept.transpose(0, 2, 1) @ X_neighbors_with_intercept
    XTy_all = (
        X_neighbors_with_intercept.transpose(0, 2, 1) @ y_neighbors[..., np.newaxis]
    ).squeeze(-1)

    # Step 5: Solve linear systems for all points simultaneously
    # Add small regularization for numerical stability (matching LinearRegression behavior)
    d_with_intercept = d + 1
    regularization_scale = 1e-10
    trace_per_point = np.trace(XTX_all, axis1=1, axis2=2)  # (n,)
    identity_matrix = np.eye(d_with_intercept)  # (d+1, d+1)
    regularization = (
        regularization_scale
        * trace_per_point[:, np.newaxis, np.newaxis]
        * identity_matrix[np.newaxis, :, :]
        / d_with_intercept
    )
    XTX_regularized = XTX_all + regularization

    # Solve: (X^T X + reg) @ beta = X^T y for all points
    XTy_expanded = XTy_all[..., np.newaxis]  # (n, d+1, 1) for broadcasting
    beta_all = np.linalg.solve(XTX_regularized, XTy_expanded).squeeze(-1)  # (n, d+1)

    # Step 6: Make predictions for all points using their local models
    # Each point uses its own local model coefficients
    X_with_intercept = np.concatenate([np.ones((n, 1)), X], axis=1)  # (n, d+1)
    y_pred_all = np.einsum("ij,ij->i", X_with_intercept, beta_all)  # (n,)

    # Step 7: Compute residuals (distance from actual to predicted)
    residuals = np.abs(y - y_pred_all)  # (n,)

    # Step 8: Compute local scale (MAD) for normalization
    local_medians = np.median(y_neighbors, axis=1)  # (n,)
    local_mads = np.median(
        np.abs(y_neighbors - local_medians[:, np.newaxis]), axis=1
    )  # (n,)

    # Precompute global MAD as fallback
    y_mean = np.mean(y)
    global_mad = np.median(np.abs(y - y_mean))

    # Step 9: Normalize residuals by local scale (with global fallback)
    epsilon = 1e-10
    use_local_scale = local_mads > epsilon
    nu = np.where(
        use_local_scale,
        residuals / (local_mads + epsilon),
        residuals / (global_mad + epsilon),
    )

    # Step 10: Transform nonlinearity scores to weights
    nu_median = np.median(nu)
    nu_mad = np.median(np.abs(nu - nu_median))

    # If all values are similar, return uniform weights
    if nu_mad < epsilon:
        return np.ones(n) / n

    # Clip extreme values, shift, and apply power transformation
    clip_range = 2.5
    shift_amount = 1.5
    power = 0.75
    nu_clipped = np.clip(
        nu, nu_median - clip_range * nu_mad, nu_median + clip_range * nu_mad
    )
    nu_shifted = nu_clipped - nu_median + shift_amount * nu_mad
    nu_powered = np.maximum(nu_shifted, epsilon) ** power

    # Step 11: Normalize to form probability weights (sum = 1)
    weights = nu_powered / np.sum(nu_powered)
    return weights
