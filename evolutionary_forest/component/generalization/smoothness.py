import numpy as np


def function_second_order_smoothness(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    delta_y = np.diff(np.diff(y))
    return np.mean((delta_y) ** 2)


def function_first_order_smoothness(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    delta_y = np.diff(y)
    return np.mean((delta_y) ** 2)


def function_first_order_smoothness_difference(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    delta_y = np.diff((y))
    delta_y_truth = np.diff((y_truth))
    return np.mean((delta_y - delta_y_truth) ** 2)


def function_second_order_smoothness_difference(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    delta_y = np.diff(np.diff(y))
    delta_y_truth = np.diff(np.diff(y_truth))
    return np.mean((delta_y - delta_y_truth) ** 2)


def function_first_order_relative_smoothness(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(y)
    delta_y_truth = np.diff(y_truth)
    delta_y_truth[abs(delta_y_truth) < 1e-5] = 1e-5
    return np.mean((delta_y / delta_y_truth) ** 2)


def function_second_order_relative_smoothness(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(np.diff(y))
    delta_y_truth = np.diff(np.diff(y_truth))
    delta_y_truth[abs(delta_y_truth) < 1e-5] = 1e-5
    return np.mean((delta_y / delta_y_truth) ** 2)


if __name__ == "__main__":
    y = np.array([2, 3, 2, 5, 6])
    y_truth = np.array([2, 3, 2, 1, 6])
