import numpy as np


def compute_T(xi, xj, fmax, fmin):
    numerator = np.sum(np.maximum(0, (xj - xi) / (fmax - fmin)))
    denominator = np.sum(np.maximum(0, (xi - xj) / (fmax - fmin)))

    if denominator == 0:  # To prevent division by zero.
        return float("inf")

    return numerator / denominator


def mu(xi, non_dominated):
    fmax = np.max(non_dominated, axis=0)
    fmin = np.min(non_dominated, axis=0)
    Ts = [compute_T(xi, xj, fmax, fmin) for xj in non_dominated]
    return min(Ts)


def knee_point_by_utility(non_dominated):
    mus = [(index, xi, mu(xi, non_dominated)) for index, xi in enumerate(non_dominated)]

    # Sort by the mu values in descending order and pick the first item which will have the largest mu value.
    best_index, best_sol, largest_mu = sorted(
        mus, key=lambda item: item[2], reverse=True
    )[0]

    return best_index, best_sol, largest_mu
