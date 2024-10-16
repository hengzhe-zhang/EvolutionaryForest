import numpy as np
from numpy.linalg import eig, lstsq


def solve_transformation_matrix(phi_X, D, p=None, regularization=1e-5, verbose=False):
    """
    Solves for the transformation matrix W that minimizes the loss ||D - D'||^2,
    where D' is the distance matrix in the transformed space defined by W.

    Parameters:
    ----------
    phi_X : np.ndarray
        The transformed feature matrix of shape (n_samples, k).
    D : np.ndarray
        The original distance matrix based on labels, of shape (n_samples, n_samples).
    p : int, optional
        The desired dimensionality of the transformed space. If None, p = k.
    regularization : float, optional
        Regularization parameter to stabilize the least squares solution.
    verbose : bool, optional
        If True, prints detailed debug information.

    Returns:
    -------
    W : np.ndarray
        The transformation matrix of shape (k, p) that maps the original features
        to the transformed space.
    """

    n, k = phi_X.shape

    if D.shape != (n, n):
        raise ValueError(
            "The distance matrix D must be of shape (n_samples, n_samples)."
        )

    if p is None:
        p = k
    elif p > k:
        raise ValueError(
            "Desired dimensionality p cannot be greater than the feature dimension k."
        )

    # Step 1: Construct matrix B and vector d
    # Each row of B corresponds to vec( (phi_i - phi_j)(phi_i - phi_j)^T )
    # and each element of d corresponds to D(i, j)

    # Initialize lists to store B rows and d elements
    B_rows = []
    d_elements = []

    if verbose:
        print("Constructing matrix B and vector d...")
    for i in range(n):
        for j in range(n):
            diff = phi_X[i] - phi_X[j]  # Shape: (k,)
            # Outer product to get (k, k) matrix, then vectorize to (k^2,)
            outer_product = np.outer(diff, diff).reshape(-1)
            B_rows.append(outer_product)
            d_elements.append(D[i, j])

    B = np.array(B_rows)  # Shape: (n^2, k^2)
    d = np.array(d_elements)  # Shape: (n^2,)

    if verbose:
        print(f"Matrix B shape: {B.shape}")
        print(f"Vector d shape: {d.shape}")

    # Step 2: Solve the least squares problem with regularization
    # m = (B^T B + reg * I)^(-1) B^T d
    # To improve numerical stability, add regularization
    if verbose:
        print("Solving the least squares problem...")
    BTB = B.T @ B  # Shape: (k^2, k^2)
    BTd = B.T @ d  # Shape: (k^2,)

    # Add regularization to the diagonal
    BTB += regularization * np.eye(BTB.shape[0])

    # Solve for m
    # Using numpy.linalg.lstsq for better numerical stability
    # Alternatively, you can use np.linalg.solve if BTB is guaranteed to be invertible
    m, residuals, rank, s = lstsq(BTB, BTd, rcond=None)

    # Step 3: Reshape m to form matrix M
    M = m.reshape(k, k)

    # Ensure M is symmetric
    M = (M + M.T) / 2

    if verbose:
        print("Matrix M obtained from least squares solution.")

    # Step 4: Eigen-decomposition of M to retrieve W
    if verbose:
        print("Performing eigen-decomposition on M...")
    eigenvalues, eigenvectors = eig(M)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top p eigenvalues and corresponding eigenvectors
    U_p = eigenvectors[:, :p]
    Lambda_p = np.diag(np.sqrt(np.maximum(eigenvalues[:p], 0)))  # Ensure non-negative

    # Compute W
    W = U_p @ Lambda_p

    if verbose:
        print(f"Transformation matrix W of shape {W.shape} obtained.")

    return W.real  # Return the real part in case of numerical complex values


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
            D[i, j] = np.abs(y[i] - y[j])  # Example: binary labels distance

    # Solve for W with verbose output
    W = solve_transformation_matrix(phi_X, D, p=2, verbose=True)

    print("Transformation matrix W:")
    print(W)
