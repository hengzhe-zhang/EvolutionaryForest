import numpy as np
from scipy.stats import pearsonr


def compute_correlation_matrix(X):
    """Compute the correlation matrix for the dataset X."""
    n_features = X.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(i, n_features):
            if i == j:
                corr_matrix[i, j] = 1
            else:
                corr_matrix[i, j], _ = pearsonr(X[:, i], X[:, j])
                corr_matrix[j, i] = corr_matrix[i, j]  # Symmetric property
    return corr_matrix


def find_most_redundant_feature(X):
    """Find and return the index of the most redundant feature in the dataset X."""
    corr_matrix = compute_correlation_matrix(X)
    np.fill_diagonal(corr_matrix, 0)  # Fill diagonal with 0s to ignore self-correlation
    redundancy_scores = abs(corr_matrix).max(
        axis=1
    )  # Max correlation for each feature with others
    most_redundant_feature_idx = (
        redundancy_scores.argmax()
    )  # Index of the most redundant feature
    return most_redundant_feature_idx
