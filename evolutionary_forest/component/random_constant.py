import numpy as np


def scaled_random_constant(biggest_val):
    """
    Generates a random constant value within a range that's scaled based
    on the maximum absolute value of the dataset.
    """
    min_erc = -5 * biggest_val
    max_erc = 5 * biggest_val
    generator = (
        lambda: round((np.random.rand() * (max_erc - min_erc) + min_erc) * 1e3) / 1e3
    )
    return generator
