import numpy as np


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
