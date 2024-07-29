import numpy as np
from matplotlib import pyplot as plt


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


def function_first_order_smoothness(y, y_truth):
    index = np.argsort(y_truth)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(y)
    smoothness_a = np.mean((delta_y) ** 2)

    index = np.argsort(y)
    y = y[index]
    y_truth = y_truth[index]
    delta_y = np.diff(y_truth)
    smoothness_b = np.mean((delta_y) ** 2)
    return min(smoothness_a, smoothness_b)


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
    # y = np.cos(np.arange(-50, 50) * 0.1 + math.pi / 2)
    y_truth = np.arange(-50, 50) ** 2 * 0.001
    # y_truth = np.sin(np.arange(-50, 50) * 0.5)
    # y = (
    #     LinearRegression()
    #     .fit(np.arange(-50, 50).reshape(-1, 1), y_truth)
    #     .predict(np.arange(-50, 50).reshape(-1, 1))
    # )
    # y = np.arange(-50, 50) * -0.01
    y = np.sin(np.arange(-50, 50) * 5 + 0.5) * 0.3
    # y = np.zeros(100)
    # y = np.sin(np.arange(-50, 50) * 0.1) + np.random.normal(0, 0.2, 100)
    # y = np.sin(np.arange(-50, 50) * 0.1) + np.random.laplace(0, 0.2, 100)
    # y = np.sin(np.arange(-50, 50) ** 2)
    # y_truth = np.arange(0, 100) * 0.1
    # y, y_truth = y_truth, y

    plt.plot(y, label="Predicted")
    plt.plot(y_truth, label="True Values")
    plt.legend()
    plt.show()
    # print(function_first_order_smoothness_difference(y, y_truth))
    # print(function_first_order_smoothness_difference(y_truth, y))
    print(function_first_order_smoothness(y, y_truth))
    plt.plot(y_truth, label="Predicted")
    plt.plot(y, label="True Values")
    plt.legend()
    plt.show()
    print(function_first_order_smoothness(y_truth, y))
