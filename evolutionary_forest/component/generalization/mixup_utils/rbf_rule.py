import numpy as np


def silvermans_rule_of_thumb_gamma(X):
    """
    Compute the gamma value for the RBF kernel using Silverman's Rule of Thumb.

    Parameters:
    X (numpy.ndarray): The input data array of shape (n_samples, n_features).

    Returns:
    float: The gamma value for the RBF kernel.
    """
    n_samples, n_features = X.shape
    std_dev = np.std(X, axis=0).mean()

    # Apply Silverman's Rule of Thumb for multivariate data
    factor = (4 / (n_features + 2)) ** (1 / (n_features + 4))
    sigma = factor * n_samples ** (-1 / (n_features + 4)) * std_dev

    # Convert sigma to gamma
    gamma = 1 / (2 * sigma**2)
    return gamma


def scotts_rule_gamma(X):
    """
    Compute the gamma value for the RBF kernel using Scott's Rule.

    Parameters:
    X (numpy.ndarray): The input data array of shape (n_samples, n_features).

    Returns:
    float: The gamma value for the RBF kernel.
    """
    n_samples, n_features = X.shape
    std_dev = np.std(X, axis=0).mean()

    # Apply Scott's Rule for multivariate data
    sigma = n_samples ** (-1 / (n_features + 4)) * std_dev

    # Convert sigma to gamma
    gamma = 1 / (2 * sigma**2)
    return gamma


if __name__ == "__main__":
    X = np.random.rand(100, 2)
    # gamma = silvermans_rule_of_thumb_gamma(X)
    gamma = scotts_rule_gamma(X)
    print(f"Gamma value: {gamma}")
