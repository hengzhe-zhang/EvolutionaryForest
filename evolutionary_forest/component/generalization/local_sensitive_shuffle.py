import numpy as np
import numba


@numba.jit(nopython=True)
def local_sensitive_shuffle_by_value(vector, variability_scale=1.0):
    """
    Shuffles elements in the vector that are within a certain range of standard deviation from each other,
    using Numba for acceleration.

    Args:
    - vector (np.ndarray): The original vector to be shuffled.
    - variability_scale (float): Defines the range within which elements can be considered for shuffling,
                                 expressed as a multiple of the standard deviation.

    Returns:
    - np.ndarray: A new vector with elements shuffled within the specified variability range.
    """
    std = np.std(vector)  # Compute the standard deviation of the vector
    shuffled_vector = vector.copy()  # Copy the vector to preserve the original

    for i in range(vector.shape[0]):
        value = vector[i]
        # Find indices of elements within the variability range from the current value
        eligible_indices = np.where(np.abs(vector - value) <= variability_scale * std)[
            0
        ]

        # Filter out the current index to avoid swapping an element with itself
        eligible_indices = eligible_indices[eligible_indices != i]

        if eligible_indices.size > 0:
            # Randomly select an index from the eligible indices for swapping
            swap_with = eligible_indices[np.random.randint(eligible_indices.size)]

            # Swap the current element with the selected element
            shuffled_vector[i], shuffled_vector[swap_with] = (
                shuffled_vector[swap_with],
                shuffled_vector[i],
            )

    return shuffled_vector


if __name__ == "__main__":
    # Example usage
    original_vector = np.random.rand(20)  # Generate a random vector of 20 elements
    variability_scale = 1.0  # Define the variability scale as 1 standard deviation

    shuffled_vector = local_sensitive_shuffle_by_value(
        original_vector, variability_scale
    )
    print("Original Vector:", original_vector)
    print("Shuffled Vector:", shuffled_vector)
