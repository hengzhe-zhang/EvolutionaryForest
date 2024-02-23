import numpy as np


def discretize_vector(vector, bins):
    """Discretize a vector into equal width bins."""
    min_val, max_val = np.min(vector), np.max(vector)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    digitized = np.digitize(vector, bin_edges, right=True)
    # Handle the case where a value is exactly equal to max_val
    digitized[digitized > bins] = bins
    return digitized


def calculate_histogram(vector, bins):
    """Calculate normalized histogram for a discretized vector."""
    histogram, _ = np.histogram(vector, bins=range(1, bins + 2))
    return histogram / np.sum(histogram)


def cross_entropy(p, q):
    """Calculate cross entropy between two probability distributions."""
    return -np.sum(p * np.log(q + 1e-9))  # 1e-9 for numerical stability


def calculate_cross_entropy(vector1, vector2, bins):
    """Calculate cross entropy between two continuous vectors."""
    # Discretize vectors
    disc_vector1 = discretize_vector(vector1, bins)
    disc_vector2 = discretize_vector(vector2, bins)

    # Calculate normalized histograms as probability distributions
    p = calculate_histogram(disc_vector1, bins)
    q = calculate_histogram(disc_vector2, bins)

    # Compute cross entropy
    return cross_entropy(p, q)


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
