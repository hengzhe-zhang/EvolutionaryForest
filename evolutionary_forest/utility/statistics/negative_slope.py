import numpy as np


def nsc_log(offspring, nsc_log_list):
    pair = []
    for ind in offspring:
        if not hasattr(ind, 'parent_fitness') or ind.parent_fitness is None:
            return
        # NSC is for minimization problems
        pair.append((-1 * ind.parent_fitness[0], -1 * ind.fitness.wvalues[0]))
    nsc_log_list.append(calculate_nsc(pair))


def calculate_nsc(parent_child_pairs, num_bins=10):
    """
    Fast, vectorized version to calculate the Negative Slope Coefficient (NSC).

    Args:
        parent_child_pairs (list or np.ndarray): (parent_fitness, child_fitness) pairs
        num_bins (int): Number of bins to split parent fitness space into.

    Returns:
        float: NSC value (sum of slopes between bin averages).
    """
    data = np.asarray(parent_child_pairs)

    if len(data) < num_bins + 1:
        raise ValueError(f"Need at least {num_bins + 1} data points for {num_bins} bins.")

    # Sort by parent fitness (column 0)
    sorted_indices = np.argsort(data[:, 0])
    sorted_data = data[sorted_indices]

    # Split into bins
    bins = np.array_split(sorted_data, num_bins)

    # Compute means for each bin in a vectorized way
    bin_means = np.array([
        [np.mean(b[:, 0]), np.mean(b[:, 1])] for b in bins if len(b) > 0
    ])

    # Compute differences between successive bin means
    delta = np.diff(bin_means, axis=0)
    dx = delta[:, 0]
    dy = delta[:, 1]

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.divide(dy, dx)
        slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]  # remove invalid slopes

    # NSC is the sum of slopes
    return np.sum(slopes)


if __name__ == '__main__':
    pairs = [
        (0.4, 0.38), (0.5, 0.48), (0.6, 0.62),
        (0.7, 0.65), (0.8, 0.85), (0.9, 0.95),
        (1.0, 0.9), (1.1, 1.2), (1.2, 1.15),
        (1.3, 1.25), (1.4, 1.35)
    ]

    nsc_value = calculate_nsc(pairs, num_bins=5)
    print(f"NSC: {nsc_value:.4f}")
