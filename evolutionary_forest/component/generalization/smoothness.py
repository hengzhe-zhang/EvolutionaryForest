import numpy as np
from deap.gp import PrimitiveTree
from matplotlib import pyplot as plt


def llm_complexity(tree: PrimitiveTree, semantics: np.ndarray):
    complexity_dict = {
        "Add": 1,
        "Sub": 1,
        "Mul": 2,
        "AQ": 5,
        "Sqrt": 3,
        "AbsLog": 4,
        "Abs": 2,
        "Square": 2,
        "Max": 2,
        "Min": 2,
        "Neg": 1,
        "SinPi": 4,
        "CosPi": 4,
    }
    score = 0
    for node in tree:
        score += complexity_dict.get(node.name, 1)
    return score


def calculate_pearson_correlation(x, y):
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the numerator and denominator of the Pearson formula
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    # Avoid division by zero
    if denominator == 0:
        return 0.0

    return numerator / denominator


def smoothness(semantics: np.ndarray, target, x):
    # normalize semantics, because of linear scaling
    semantics = semantics / np.linalg.norm(semantics)
    target = target / np.linalg.norm(target)

    smoothness = np.inf
    for x_i in x.T:
        idx = np.argsort(x_i)
        smoothness = min(
            smoothness, np.mean((np.diff(semantics[idx]) - np.diff(target[idx]) ** 2))
        )
    return smoothness


def function_second_order_smoothness(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(np.diff(y))
    smoothness_a = np.mean((delta_y) ** 2)

    index = np.argsort(y)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(np.diff(y_truth))
    smoothness_b = np.mean((delta_y) ** 2)
    return min(smoothness_a, smoothness_b)


def function_first_order_variance(y, y_truth=None):
    index = np.argsort(y)
    y = y[index]
    delta_y = np.diff(y)
    return np.mean((delta_y) ** 2)


def function_first_order_smoothness(y, y_truth, plot=False, average_version=False):
    if average_version:
        y = mean_of_parts(np.asarray(y))
        y_truth = mean_of_parts(np.asarray(y_truth))
    index = np.argsort(y_truth)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(y)
    smoothness_a = np.mean((delta_y) ** 2)

    # Plotting if required
    if plot:
        plt.figure(figsize=(12, 6))

        # Plot original sequences
        plt.subplot(1, 2, 1)
        plt.plot(y, label="Sorted y")
        plt.plot(y_truth, label="Sorted y_truth")
        plt.title("Original Sequences")
        plt.legend()

        # Plot smoothed sequences
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(y) - 1), np.diff(y), label="First Differences of y")
        plt.plot(
            np.arange(len(y_truth) - 1),
            np.diff(y_truth),
            label="First Differences of y_truth",
        )
        plt.title("First Differences")
        plt.legend()

        plt.tight_layout()
        plt.show()

    index = np.argsort(y)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(y_truth)
    smoothness_b = np.mean((delta_y) ** 2)

    return min(smoothness_a, smoothness_b)


def mean_of_parts(arr, num_parts=5):
    if len(arr) % num_parts != 0:
        raise ValueError(
            "The length of the array must be divisible by the number of parts."
        )
    reshaped_arr = arr.reshape(-1, num_parts)
    return np.mean(reshaped_arr, axis=1)


def function_first_order_smoothness_difference(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff((y))
    delta_y_truth = np.diff(y_truth)
    smoothness_a = np.mean((delta_y - delta_y_truth) ** 2)
    index = np.argsort(y)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff((y))
    delta_y_truth = np.diff(y_truth)
    smoothness_b = np.mean((delta_y - delta_y_truth) ** 2)
    return min(smoothness_a, smoothness_b)
    # return smoothness_a


def function_second_order_smoothness_difference(y, y_truth):
    # y, y_truth = copy.deepcopy(y), copy.deepcopy(y_truth)
    index = np.argsort(y_truth)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(np.diff(y))
    delta_y_truth = np.diff(np.diff(y_truth))
    smoothness_a = np.mean((delta_y - delta_y_truth) ** 2)
    # plt.plot(delta_y)
    # plt.plot(delta_y_truth)
    # plt.show()
    index = np.argsort(y)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(np.diff(y))
    delta_y_truth = np.diff(np.diff(y_truth))
    # plt.plot(delta_y)
    # plt.plot(delta_y_truth)
    # plt.show()
    smoothness_b = np.mean((delta_y - delta_y_truth) ** 2)
    return min(smoothness_a, smoothness_b)
    # return max(smoothness_a, smoothness_b)
    # return smoothness_a


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
    # Define different y configurations
    y_configs = [
        np.sin(np.arange(-50, 50) * 5 + 0.5) * 0.3,
        np.zeros(100),
        np.sin(np.arange(-50, 50) * 0.1) + np.random.normal(0, 0.2, 100),
        np.sin(np.arange(-50, 50) * 0.1) + np.random.laplace(0, 0.2, 100),
        np.sin(np.arange(-50, 50) ** 2),
    ]

    # Define the y_truth (can be modified as needed)
    y_truth = np.arange(-50, 50) ** 2 * 0.001

    # Loop through different y configurations
    for i, y in enumerate(y_configs):
        plt.figure()
        plt.plot(y, label="Predicted")
        plt.plot(y_truth, label="True Values")
        plt.legend()
        plt.title(f"Configuration {i + 1}")
        plt.show()

        # Calculate smoothness
        smoothness_y = function_first_order_variance(y, y_truth)
        print(f"Configuration {i + 1}:")
        print(f"  First-order smoothness of y: {smoothness_y}")
        print()
