import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from numpy.linalg import eig


def compute_lambda_matrix(y):
    """Compute RBF weight matrix for targets Y using γ = 1 / Var(Y)."""
    y = np.asarray(y).reshape(-1, 1)
    gamma = 1 / np.var(y)
    return rbf_kernel(y, gamma=gamma)


def safe_lstsq(A, b, rcond=None):
    """
    Safe least squares when A is already regularized/conditioned
    """
    try:
        # Try standard lstsq first
        result = np.linalg.lstsq(A, b, rcond=rcond)
        return result
    except np.linalg.LinAlgError:
        # Fallback: Use pseudoinverse (no additional regularization)
        A_pinv = np.linalg.pinv(A, rcond=rcond)
        m = A_pinv @ b
        residuals = (
            np.sum((A @ m - b) ** 2) if A.shape[0] >= A.shape[1] else np.array([])
        )
        rank = np.linalg.matrix_rank(A)
        s = np.array([])  # Skip SVD computation to avoid same convergence issue
        return m, residuals, rank, s


def compute_laplacian_term(phi_X, y, sigma=1.0):
    n = phi_X.shape[0]

    # Compute label-based similarity matrix S
    # S_ij = exp(-(y_i - y_j)^2 / (2 * sigma^2))
    y_diff = y[:, np.newaxis] - y[np.newaxis, :]
    S = np.exp(-(y_diff**2) / (2 * sigma**2))

    # Compute graph Laplacian L = D - S
    D_degree = np.diag(S.sum(axis=1))  # Degree matrix
    L = D_degree - S  # Laplacian matrix

    # Compute Phi^T L Phi
    manifold_matrix = phi_X.T @ L @ phi_X  # Shape: (k, k)

    # Vectorize to get g = vec(Phi^T L Phi)
    g = manifold_matrix.flatten()  # Shape: (k^2,)

    return g


def solve_transformation_matrix(
    phi_X,
    D,
    y=None,
    p=None,
    weights=None,
    regularization=1e-5,
    laplacian_reg=0.0,
    laplacian_sigma=0.1,
):
    """
    Solves for the transformation matrix W that minimizes the loss ||D - D'||^2,
    with optional Laplacian regularization.

    Parameters:
    ----------
    phi_X : np.ndarray
        The transformed feature matrix of shape (n_samples, k).
    D : np.ndarray
        The original distance matrix based on labels, of shape (n_samples, n_samples).
    y : np.ndarray, optional
        The target labels of shape (n_samples,). Required if laplacian_reg > 0.
    p : int, optional
        The desired dimensionality of the transformed space. If None, p = k.
    weights : np.ndarray, optional
        A weight matrix of shape (n_samples, n_samples) for the contrastive loss.
        If None, all pairs are equally weighted.
    regularization : float, optional
        L2 regularization parameter (lambda_1) to stabilize the solution.
    laplacian_reg : float, optional
        Laplacian regularization parameter (lambda_2). If 0, no Laplacian regularization is applied.
    laplacian_sigma : float, optional
        Bandwidth parameter for the RBF kernel in Laplacian computation.

    Returns:
    -------
    W : np.ndarray
        The transformation matrix of shape (k, p).
    """

    n, k = phi_X.shape

    if D.shape != (n, n):
        raise ValueError(
            "The distance matrix D must be of shape (n_samples, n_samples)."
        )

    if y is None and laplacian_reg > 0:
        raise ValueError(
            "Labels y must be provided when using Laplacian regularization."
        )

    if p is None:
        p = k
    elif p > k:
        raise ValueError(
            "Desired dimensionality p cannot be greater than the feature dimension k."
        )

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != (n, n):
            raise ValueError(
                "Weights must be a matrix with shape (n_samples, n_samples)."
            )
    else:
        weights = np.ones((n, n))

    # Step 1: Construct matrix B and vector d for contrastive loss
    B_rows = []
    d_elements = []
    weight_factors = []

    for i in range(n):
        for j in range(n):
            diff = phi_X[i] - phi_X[j]
            outer_product = np.outer(diff, diff).reshape(-1)
            B_rows.append(outer_product)
            d_elements.append(D[i, j])
            weight_factors.append(weights[i, j])

    B = np.array(B_rows)  # Shape: (n^2, k^2)
    d = np.array(d_elements)  # Shape: (n^2,)
    weight_factors = np.array(weight_factors)  # Shape: (n^2,)

    # Apply weights to B and d
    if not np.all(weights == 1):
        sqrt_weights = np.sqrt(weight_factors)
        B = B * sqrt_weights[:, np.newaxis]
        d = d * sqrt_weights

    # Step 2: Compute Laplacian regularization term (if enabled)
    g = None
    if laplacian_reg > 0:
        g = compute_laplacian_term(phi_X, y, sigma=laplacian_sigma)

    # Step 3: Solve the regularized least squares problem
    BTB = B.T @ B  # Shape: (k^2, k^2)
    BTd = B.T @ d  # Shape: (k^2,)

    # Add L2 regularization to the diagonal
    BTB += regularization * np.eye(BTB.shape[0])

    # Construct right-hand side
    if laplacian_reg > 0 and g is not None:
        rhs = BTd - (laplacian_reg / 2) * g
    else:
        rhs = BTd

    # Solve for m
    m, residuals, rank, s = safe_lstsq(BTB, rhs, rcond=None)

    # Step 4: Reshape m to form matrix M
    M = m.reshape(k, k)

    # Ensure M is symmetric
    M = (M + M.T) / 2

    # Step 5: Eigen-decomposition of M to retrieve W
    eigenvalues, eigenvectors = eig(M)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top p eigenvalues and corresponding eigenvectors
    U_p = eigenvectors[:, :p]
    # Ensure eigenvalues are non-negative before taking the square root
    Lambda_p = np.diag(np.sqrt(np.maximum(eigenvalues[:p], 0)))

    # Compute W
    W = U_p @ Lambda_p

    return W.real


# Example Usage
if __name__ == "__main__":
    # Example data
    np.random.seed(0)
    n_samples = 5
    k_features = 3

    # Random feature matrix phi_X
    phi_X = np.random.rand(n_samples, k_features)

    # Compute original distance matrix D based on some labels y
    # For illustration, we'll compute Euclidean distances based on random labels
    y = np.random.randint(0, 2, size=n_samples)
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            D[i, j] = (y[i] - y[j]) ** 2  # Example: binary labels distance

    # Solve for W with verbose output
    W = solve_transformation_matrix(
        phi_X, D, p=2, weights=compute_lambda_matrix(y), verbose=True
    )

    print("Transformation matrix W:")
    print(W)
