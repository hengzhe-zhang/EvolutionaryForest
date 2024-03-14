import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
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


def find_most_redundant_feature(X, inverse=False):
    """Find and return the index of the most redundant feature in the dataset X."""
    corr_matrix = compute_correlation_matrix(X)
    np.fill_diagonal(corr_matrix, 0)  # Fill diagonal with 0s to ignore self-correlation
    # Max correlation for each feature with others
    redundancy_scores = abs(corr_matrix).max(axis=1)
    # Index of the most redundant feature
    most_redundant_feature_idx = redundancy_scores.argmax()
    if inverse:
        most_redundant_feature_idx = redundancy_scores.argmin()
    return most_redundant_feature_idx


def compute_model_performance(X, y):
    """
    Fits a linear model using least squares and returns the model's performance.
    Performance could be measured by the residual sum of squares (RSS) for simplicity.
    """
    # Add a column of ones to X for the intercept
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

    # Solve the least squares problem
    coefs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    # If residuals is empty, compute manually
    if len(residuals) == 0:
        predictions = X_with_intercept @ coefs
        residuals = np.sum((y - predictions) ** 2)
    return residuals if isinstance(residuals, float) else residuals.item()


def plot_feature_correlations(X, y, feature_names=None):
    """
    Plots the correlation between each feature in X and the target variable y.

    Parameters:
    X (array-like): The input features with shape (n_samples, n_features).
    y (array-like): The target variable with shape (n_samples,).
    feature_names (list of str, optional): Names of the features for x-axis labels.
    """
    correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])

    plt.figure(figsize=(10, 5))
    x_ticks = range(X.shape[1])

    if feature_names is not None:
        plt.bar(x_ticks, correlations, color="skyblue", tick_label=feature_names)
        plt.xticks(
            rotation=45, ha="right"
        )  # Rotate feature names for better readability
    else:
        plt.bar(x_ticks, correlations, color="skyblue")

    plt.xlabel("Feature")
    plt.ylabel("Correlation with Target")
    plt.title("Feature Correlation with Target")
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()


def find_least_useful_feature(X, y, inverse=False):
    """
    Uses backward elimination to determine the least useful feature.
    """
    original_performance = compute_model_performance(X, y)
    performances = np.zeros(X.shape[1])  # Initialize as a NumPy array

    for i in range(X.shape[1]):
        # Try removing each feature and compute the performance of the model without it
        X_reduced = np.delete(X, i, axis=1)
        performance_without_feature = compute_model_performance(X_reduced, y)
        performances[i] = performance_without_feature - original_performance

    plot_correlations = False  # Set to True to visualize feature correlations
    # Optionally plot the correlation between each feature and the target
    if plot_correlations:
        plot_feature_correlations(X, y)

    # The least useful feature is the one whose removal causes the least drop in performance
    least_useful_idx = np.argmin(performances)
    if inverse:
        least_useful_idx = np.argmax(performances)
    return least_useful_idx


def find_largest_p_value_feature(X, y, inverse=False):
    """
    Fits a linear model using OLS and finds the feature with the largest p-value.

    Parameters:
    X (array-like): The input features with shape (n_samples, n_features).
    y (array-like): The target variable with shape (n_samples,).

    Returns:
    int: The index of the feature with the largest p-value, suggesting it's the least significant.
    """
    # Add a constant to the features for the intercept
    X_with_intercept = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X_with_intercept).fit()

    # Obtain the p-values for the coefficients (excluding the intercept)
    p_values = model.pvalues[1:]  # pvalues[0] corresponds to the intercept

    # Find the index of the feature with the largest p-value
    largest_p_value_idx = np.argmax(p_values)

    if inverse:
        largest_p_value_idx = np.argmin(p_values)
    return largest_p_value_idx


def find_least_correlation_to_target(X, y, inverse=False):
    """
    Finds the index of the feature in X that has the least correlation (absolute value) with the target variable y.

    Parameters:
    X (array-like): The input features with shape (n_samples, n_features).
    y (array-like): The target variable with shape (n_samples,).

    Returns:
    int: The index of the feature with the least absolute correlation with the target.
    """
    # Initialize an array to hold the correlation values
    correlations = np.zeros(X.shape[1])

    # Compute the correlation of each feature with the target
    for i in range(X.shape[1]):
        correlations[i] = np.corrcoef(X[:, i], y)[0, 1]

    # Find the index of the feature with the smallest absolute correlation
    least_corr_idx = np.argmin(np.abs(correlations))
    if inverse:
        least_corr_idx = np.argmax(np.abs(correlations))

    return least_corr_idx
