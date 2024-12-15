import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

from evolutionary_forest.utility.sampling_utils import sample_according_to_distance


def mixup_regression(X, y, alpha=0.2, gamma_value_in_kernel=1.0):
    """
    Perform MixUp data augmentation for regression tasks with distance-based sampling.

    Parameters:
    - X (numpy.ndarray): Input features, shape (n_samples, n_features).
    - y (numpy.ndarray): Target values, shape (n_samples,).
    - alpha (float): MixUp interpolation strength. Must be > 0.
    - gamma_value_in_kernel (float): Gamma parameter for the RBF kernel.

    Returns:
    - X_mix (numpy.ndarray): Augmented features.
    - y_mix (numpy.ndarray): Augmented targets.
    """
    if alpha <= 0:
        raise ValueError("Alpha must be greater than 0")

    # Number of samples
    n_samples = X.shape[0]

    # Generate lambda from Beta distribution
    lam = np.random.beta(alpha, alpha, size=n_samples)

    # Compute distance matrix using RBF kernel
    XY = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    distance_matrix = rbf_kernel(XY, gamma=gamma_value_in_kernel)

    # Define indices for mixup
    indices_a = np.arange(n_samples)  # Original indices
    indices_b = sample_according_to_distance(
        distance_matrix, indices_a
    )  # Sample based on distance

    # Perform mixup
    X_mix = lam[:, None] * X[indices_a] + (1 - lam)[:, None] * X[indices_b]
    y_mix = lam * y[indices_a] + (1 - lam) * y[indices_b]

    return X_mix, y_mix.flatten()


def data_augmentation(X, y, alpha=0.2, gamma_value_in_kernel=1.0):
    X_mix, y_mix = mixup_regression(X, y.flatten(), alpha, gamma_value_in_kernel)
    X_aug = np.vstack((X, X_mix))
    y_aug = np.concatenate((y.flatten(), y_mix))
    return X_aug, y_aug


# Example usage
if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)

    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

    # Apply MixUp
    X_mix, y_mix = data_augmentation(X, y, alpha=0.2)

    print("Original Features:")
    print(X)
    print("Original Targets:")
    print(y)
    print("MixUp Features:")
    print(X_mix)
    print("MixUp Targets:")
    print(y_mix)
