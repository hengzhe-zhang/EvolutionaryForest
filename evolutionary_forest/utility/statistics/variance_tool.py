import numpy as np


def mean_without_outliers(data, metric="Mean"):
    """
    Calculate the standard deviation of a list/array of numbers, ignoring outliers.

    Parameters:
        data (list or numpy.array): Input data set.

    Returns:
        float: Standard deviation of the data set after removing outliers.
    """
    data = np.array(data)  # Ensure data is a numpy array for easier processing
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    if metric == "Mean":
        # Compute mean
        return np.mean(filtered_data)
    elif metric == "Median":
        # Compute median
        return np.median(filtered_data)
    else:
        # Compute standard deviation
        return np.std(filtered_data)


if __name__ == "__main__":
    # Example usage
    data = [1, 2, 2, 3, 4, 4, 4, 5, 100]  # Data with an outlier
    std_value = mean_without_outliers(data)
    print("Standard Deviation after removing outliers:", std_value)
