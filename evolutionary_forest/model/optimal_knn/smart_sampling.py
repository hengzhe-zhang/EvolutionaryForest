import numpy as np


def uniform_sampling_indices(labels, n_samples, n_bins=10, random_state=None):
    """
    Perform uniform sampling for regression problems by stratifying the labels into bins.
    Ensures exactly n_samples are returned and distributes sampling more evenly.

    Parameters:
        labels (np.ndarray): Continuous target variable of shape (n_samples,).
        n_samples (int): Total number of samples to draw.
        n_bins (int): Desired number of bins to stratify the labels.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        np.ndarray: Indices of the sampled data points.
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_labels = np.unique(labels)
    n_bins = min(
        n_bins, len(unique_labels)
    )  # Adjust n_bins if there are fewer unique labels

    # Use right=True to include the rightmost edge in the last bin
    bins = np.linspace(labels.min(), labels.max(), n_bins + 1)
    bin_indices = np.digitize(labels, bins) - 1  # Assign each label to a bin

    # Handle edge case where labels.max() == bins[-1], ensuring it falls into the last bin
    bin_indices[bin_indices == n_bins] = n_bins - 1

    # Initialize sampling allocations
    samples_per_bin = np.full(n_bins, n_samples // n_bins, dtype=int)
    remainder = n_samples % n_bins
    samples_per_bin[:remainder] += 1  # Distribute the remainder

    # Initialize sampled indices list and track remaining samples
    sampled_indices = []
    remaining_samples = 0

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_indices_in_data = np.where(bin_mask)[0]
        n_to_sample = samples_per_bin[i]

        if len(bin_indices_in_data) < n_to_sample:
            # If not enough samples in this bin, take all and track the shortfall
            sampled = bin_indices_in_data
            sampled_indices.extend(sampled)
            shortfall = n_to_sample - len(sampled)
            remaining_samples += shortfall
        else:
            # Sample without replacement
            sampled = np.random.choice(bin_indices_in_data, n_to_sample, replace=False)
            sampled_indices.extend(sampled)

    # If there are remaining samples, redistribute them to bins that can afford more samples
    if remaining_samples > 0:
        # Identify bins that still have available samples
        available_bins = []
        for i in range(n_bins):
            bin_mask = bin_indices == i
            bin_indices_in_data = np.where(bin_mask)[0]
            already_sampled = set(sampled_indices)
            available = np.setdiff1d(bin_indices_in_data, list(already_sampled))
            available_count = len(available)
            if available_count > 0:
                available_bins.append(available)

        # Flatten the list of available indices
        if available_bins:
            available_indices = np.concatenate(available_bins)
        else:
            available_indices = np.array([])

        if len(available_indices) < remaining_samples:
            raise ValueError(
                f"Not enough samples to fulfill the requested n_samples={n_samples}. "
                f"Only {len(sampled_indices) + len(available_indices)} samples available."
            )

        additional_samples = np.random.choice(
            available_indices, remaining_samples, replace=False
        )
        sampled_indices.extend(additional_samples)

    sampled_indices = np.array(sampled_indices)

    # Shuffle the sampled indices to ensure randomness
    np.random.shuffle(sampled_indices)

    return sampled_indices


if __name__ == "__main__":
    features = np.random.rand(100, 5)  # 100 samples with 5 features each
    labels = np.random.uniform(0, 100, 100)  # Continuous target variable (0 to 100)

    # Parameters
    n_samples = 50  # Total samples to draw
    n_bins = 5  # Number of bins for stratification

    # Call the function
    sampled_indices = uniform_sampling_indices(labels, n_samples, n_bins)

    # Use the indices to extract features and labels if needed
    sampled_features = features[sampled_indices]
    sampled_labels = labels[sampled_indices]

    # Display results
    print("Sampled Indices:", len(sampled_indices))
    print("Sampled Features:", len(sampled_features))
    print("Sampled Labels:", len(sampled_labels))
