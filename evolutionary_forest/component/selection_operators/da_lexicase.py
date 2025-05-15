import numpy as np


def selDAlexFast(individuals, k, particularity_pressure=3.0, normalize=True):
    """
    DALex selection on a list of individuals using weighted aggregation of errors.

    Args:
        individuals (list): Each individual must have a `case_values` attribute (array-like).
        k (int): Number of individuals to select.
        particularity_pressure (float): Std. dev. for Gaussian sampling of importance scores.
        normalize (bool): Whether to normalize case errors across individuals.

    Returns:
        list: Selected individuals.
    """
    # Extract the error matrix: shape (n_individuals, n_cases)
    case_values = np.array([ind.case_values for ind in individuals])
    E = case_values.copy()

    if normalize:
        means = E.mean(axis=0)
        stds = E.std(axis=0)
        stds[stds == 0] = 1e-8  # Prevent division by zero
        E = (E - means) / stds

    # Sample importance scores and compute softmax weights
    importance_scores = np.random.normal(
        loc=0.0, scale=particularity_pressure, size=(k, E.shape[1])
    )
    exp_scores = np.exp(
        importance_scores - np.max(importance_scores, axis=1, keepdims=True)
    )
    weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute weighted errors: shape (k, n_individuals)
    weighted_errors = weights @ E.T

    # Select indices of individuals with lowest weighted error
    selected_indices = np.argmin(weighted_errors, axis=1)

    # Return the actual selected individual objects
    return [individuals[i] for i in selected_indices]
