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


if __name__ == "__main__":
    # Example usage
    vector1 = np.random.rand(5)  # Random vector 1
    vector2 = np.random.rand(5)  # Random vector 2
    bins = 10  # Number of bins for discretization
    print(calculate_cross_entropy(vector1, vector2, bins))
