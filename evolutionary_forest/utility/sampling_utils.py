import numpy as np


def sample_according_to_distance(
    distance_matrix: np.ndarray, indices_a: np.ndarray, skip_self: bool = False
) -> np.ndarray:
    prob_distribution: np.ndarray = get_probability_matrix_from_distance_matrix(
        distance_matrix, indices_a, skip_self
    )
    # Sample indices according to the probability distribution
    indices_b: list[int] = [
        np.random.choice(len(distance_matrix), p=prob_distribution[i])
        for i in range(len(indices_a))
    ]
    return np.array(indices_b)


def get_probability_matrix_from_distance_matrix(
    distance_matrix: np.ndarray, indices_a: np.ndarray, skip_self: bool = False
) -> np.ndarray:
    """
    Sample indices according to the probability distribution given by the distance matrix.
    """
    prob_distribution: np.ndarray = distance_matrix[
        indices_a
    ]  # Extract probabilities for the given indices
    if skip_self:
        np.fill_diagonal(prob_distribution, 0)
    # Normalize to form a valid probability distribution
    prob_distribution = prob_distribution / np.sum(
        prob_distribution, axis=1, keepdims=True
    )
    return prob_distribution
