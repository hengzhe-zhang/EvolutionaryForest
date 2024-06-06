import numpy as np


def calculate_slope(x, y):
    """
    Calculate the slope (b_1) of the linear regression line.

    Parameters:
    x (numpy array): Array of feature values.
    y (numpy array): Array of target values.

    Returns:
    float: The slope (b_1).
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    b_1 = numerator / denominator
    return b_1


def calculate_intercept(x, y, b_1):
    """
    Calculate the intercept (b_0) of the linear regression line.

    Parameters:
    x (numpy array): Array of feature values.
    y (numpy array): Array of target values.
    b_1 (float): The slope of the linear regression line.

    Returns:
    float: The intercept (b_0).
    """
    y_mean = np.mean(y)
    x_mean = np.mean(x)

    b_0 = y_mean - b_1 * x_mean
    return b_0


if __name__ == "__main__":
    # Example usage
    x = np.array([1, 2, 3, 4])
    y = np.array([2, 3, 4, 5])

    b_1 = calculate_slope(x, y)
    b_0 = calculate_intercept(x, y, b_1)

    print(f"Slope (b_1): {b_1}")
    print(f"Intercept (b_0): {b_0}")
