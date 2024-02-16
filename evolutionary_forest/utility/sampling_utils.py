import numpy as np
from cachetools import cached, LRUCache
from cachetools.keys import hashkey


def sample_according_to_distance(distance_matrix, indices_a, inverse_prob=False):
    prob_distribution = get_probability_matrix_from_distance_matrix(
        distance_matrix, indices_a, inverse_prob
    )
    # Sample indices according to the probability distribution
    indices_b = [
        np.random.choice(len(distance_matrix), p=prob_distribution[i])
        for i in range(len(indices_a))
    ]
    return indices_b


def get_probability_matrix_from_distance_matrix(
    distance_matrix, indices_a, inverse_prob=False
):
    """
    Sample indices according to the probability distribution given by the distance matrix.
    """
    prob_distribution = distance_matrix[
        indices_a
    ]  # Extract probabilities for the given indices
    if inverse_prob:
        # inverse the probability vector
        prob_distribution[prob_distribution != 0] = (
            1 / prob_distribution[prob_distribution != 0]
        )
    # Normalize to form a valid probability distribution
    prob_distribution = prob_distribution / np.sum(
        prob_distribution, axis=1, keepdims=True
    )
    return prob_distribution


@cached(
    cache=LRUCache(maxsize=128),
    key=lambda data, num_samples, sigma=1.0, replace=True: hashkey(
        num_samples, sigma, replace
    ),
)
def sample_indices_gaussian_kernel(data, num_samples, sigma=1.0, replace=True):
    """
    Samples multiple indices from 'data' based on the pairwise Gaussian kernel distance to other examples.

    Parameters:
    - data: np.array, 1D array of data points.
    - num_samples: int, the number of indices to sample.
    - sigma: float, the standard deviation of the Gaussian kernel, controls the smoothness.
    - replace: bool, whether sampling is done with replacement.

    Returns:
    - sampled_indices: np.array, the sampled indices based on the Gaussian kernel distances.
    """
    # Compute pairwise squared Euclidean distances
    diff = np.expand_dims(data, 1) - np.expand_dims(data, 0)
    sq_distances = diff**2

    # Apply the Gaussian function to the squared distances
    gaussian_kernel = np.exp(-sq_distances / (2 * sigma**2))

    # Sum over rows to get the "influence" score of each point
    influence_scores = np.sum(gaussian_kernel, axis=1)

    # Normalize scores to get probabilities
    probabilities = influence_scores / np.sum(influence_scores)

    # Sample indices according to the computed probabilities
    sampled_indices = np.random.choice(
        len(data), size=num_samples, p=probabilities, replace=replace
    )

    return sampled_indices


if __name__ == "__main__":
    # Example usage:
    data = np.array([1, 2, 3])
    sapling_index = sample_indices_gaussian_kernel(data, 10)
    print("Sapling index:", sapling_index)
