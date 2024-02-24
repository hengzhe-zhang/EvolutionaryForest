import numpy as np


def discretize_and_replace(vector, bins, strategy="Min"):
    """
    Discretize a vector into bins and replace values within each bin with the min or median of the bin.
    If a bin has no values, elements corresponding to that bin are left unchanged.

    :param vector: The input vector to be discretized.
    :param bins: The number of bins to use or the bin edges.
    :param strategy: "Min" to replace with min, "Median" to replace with median within each bin.
    :return: Discretized and replaced vector.
    """
    # Discretize the vector
    bin_edges = np.histogram_bin_edges(vector, bins=bins)
    bin_indices = np.digitize(vector, bin_edges)

    # Initialize the output vector with a default value (e.g., NaN)
    discretized_vector = np.zeros_like(vector)

    # Replace values within each bin
    for i in range(1, len(bin_edges) + 1):
        # Find indices of elements in the current bin
        in_bin = bin_indices == i

        if not np.any(in_bin):
            # Skip bins with no elements
            continue

        # Calculate the replacement value (min or median) for the current bin
        if strategy == "Min":
            replacement_value = np.min(vector[in_bin])
        elif strategy == "Median":
            replacement_value = np.median(vector[in_bin])
        else:
            raise Exception("Invalid strategy")

        # Replace elements in the bin with the calculated value
        discretized_vector[in_bin] = replacement_value

    return discretized_vector


if __name__ == "__main__":
    # Example usage
    vector = np.random.rand(100)  # A random vector of 100 elements
    bins = 5  # Discretize into 5 bins
    discretized_vector_min = discretize_and_replace(
        vector, bins, strategy="Min"
    )  # Replace with min in each bin
    discretized_vector_median = discretize_and_replace(
        vector, bins, strategy="Median"
    )  # Replace with median in each bin

    print("Discretized with min:", discretized_vector_min)
    print("Discretized with median:", discretized_vector_median)
