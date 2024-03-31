import numpy as np
from sklearn.preprocessing import StandardScaler


def is_standardized(X):
    """
    Checks if a matrix X is standardized for each feature.
    A feature is considered standardized if it has a mean of 0 and a standard deviation of 1.
    This function allows for a feature to have zero variance (standard deviation of 0).
    """
    # Calculate means and standard deviations along columns (features)
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Check if all means are approximately 0
    means_condition = np.allclose(means, 0)

    # Check if all stds are 1 or, for zero variance, stds can be 0
    stds_condition = np.all(np.isclose(stds, 1) | np.isclose(stds, 0))

    return means_condition and stds_condition


if __name__ == "__main__":
    # Example matrix X
    X = np.array([[0, -1, 2], [0, 1, 0], [0, 0, 0]])
    # Check if the matrix X is standardized according to each feature
    is_standardized_flag = is_standardized(X)
    print(
        f"The matrix is standardized according to each feature: {is_standardized_flag}"
    )

    X = StandardScaler().fit_transform(X)
    # Check if the matrix X is standardized according to each feature
    is_standardized_flag = is_standardized(X)
    print(
        f"The matrix is standardized according to each feature: {is_standardized_flag}"
    )
