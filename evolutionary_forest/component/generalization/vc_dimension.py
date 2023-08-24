import copy
import math
import random

import numpy as np
from numba import njit
from sklearn.datasets import make_friedman1
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def exp_err(RMSE, h, n):
    """
    Calculates the exponential error measure given the RMSE, p, and n.

    Args:
        RMSE (float): root mean squared error of the model
        h (int): VC-dimension
        n (int): number of samples/observations

    Returns:
        float: exponential error measure
    """
    p = h / n
    term = 1 - math.sqrt(p - p * math.log(p) + (math.log(n) / (2 * n)))
    if term <= 0:
        return float('inf')
    else:
        return RMSE / term


@njit(cache=True)
def phi(tau):
    # Estimate the effective VC-dimension by observing the maximum deviation delta of error rates
    a = 0.16
    b = 1.2
    k = 0.14928

    if tau < 0.5:
        return 1
    else:
        numerator = a * (math.log(2 * tau) + 1)
        denominator = tau - k
        temp = b * (tau - k) / (math.log(2 * tau) + 1)
        radicand = 1 + temp
        return numerator / denominator * (math.sqrt(radicand) + 1)


@njit(cache=True)
def mse(h, number_of_points, all_epsilons):
    all_mse = []
    for n_samples, epsilons in zip(number_of_points, all_epsilons):
        en = (n_samples / 2) / h
        all_mse.append((phi(en) - epsilons) ** 2)
    return sum(all_mse) / len(all_mse)


@njit(cache=True)
def simple_argmin(arr):
    min_index = 0
    min_value = arr[0]
    for i in range(1, len(arr)):
        if arr[i] < min_value:
            min_index = i
            min_value = arr[i]
    return min_index


@njit(cache=True)
def vcd_mse(hs_max, number_of_points, all_epsilons):
    # Calculate VC dimension
    hs = np.arange(1, hs_max)
    results = []
    for h in hs:
        results.append(mse(h, number_of_points, all_epsilons))
    return hs[simple_argmin(results)]


def vc_dimension_estimation(X, y, estimator, input_dimension=None,estimated_vcd=None, feature_generator=None,
                            optimal_design=False):
    Yp = estimator.predict(X)
    rse = r2_score(y, Yp)
    if estimated_vcd is None:
        estimated_vcd = X.shape[1]

    estimator = copy.deepcopy(estimator)
    sequence = [0.5, 0.8, 1.0, 1.2, 2, 2.5, 3, 3.5, 5, 6.5, 8, 10, 15, 20, 30]
    assert len(sequence) == 15
    number_of_points = np.array([int(estimated_vcd * s) for s in sequence])
    max_vcd = int(estimated_vcd * sequence[-1])
    all_epsilons = []
    if input_dimension is None:
        # input dimension should be same as the number of original features
        input_dimension = X.shape[1]

    for n_samples in number_of_points:
        epsilons = []
        # Number of trials
        m = 20
        for _ in range(m):
            epsilon = maximum_deviation(n_samples, input_dimension, estimator, feature_generator)
            epsilons.append(epsilon)

        # Calculate the average maximum deviation
        all_epsilons.append(epsilons)

    all_epsilons = np.array(all_epsilons)
    h_est = vcd_mse(max_vcd, np.array(number_of_points), all_epsilons.mean(axis=1))

    if optimal_design:
        all_epsilons: list = all_epsilons.tolist()
        optimal_vcd(all_epsilons, number_of_points, max_vcd, input_dimension, estimator, feature_generator)
        # print([len(a) for a in all_epsilons])
        mean_values = np.array([np.mean(x) for x in all_epsilons if len(x) > 0])
        number_of_points_tmp = np.array([number_of_points[id] for id, x in enumerate(all_epsilons)
                                         if len(x) > 0])
        h_est = vcd_mse(max_vcd, number_of_points_tmp, mean_values)

    # return exp_err(math.sqrt(mean_squared_error(y, Yp)), h_est, len(X))
    return ((rse, 1), (h_est, -1))


def optimal_vcd(all_epsilons, number_of_points, max_vcd, dimension, estimator, feature_generator):
    mean_values = np.array([np.mean(a) for a in all_epsilons])
    h_est = vcd_mse(max_vcd, number_of_points, mean_values)
    base_mse = mse(h_est, number_of_points, mean_values)
    while True:
        # Step 1: Rank design points according to their contributions to MSE
        mse_array = []
        for exp_to_exclude in range(len(all_epsilons)):
            # Delete the specified experiment from the array
            modified_array = [a for id, a in enumerate(all_epsilons) if id != exp_to_exclude]
            # Calculate the mean value with excluding one experiment
            mean_values = np.array([np.mean(x) for x in modified_array if len(x) > 0])
            number_of_points_tmp = [p for id, p in enumerate(number_of_points) if id != exp_to_exclude]
            assert len(number_of_points_tmp) == len(all_epsilons) - 1
            number_of_points_tmp = np.array([number_of_points_tmp[id] for id, x in enumerate(modified_array)
                                             if len(x) > 0])
            assert len(number_of_points_tmp) == len(mean_values), \
                f'{len(number_of_points_tmp)} != {len(mean_values)}'

            # Get estimated VCD
            h_est = vcd_mse(max_vcd, number_of_points_tmp, mean_values)
            # Calculate MSE based on VCD
            mse_value = mse(h_est, number_of_points_tmp, mean_values)
            # Contribution of VCD
            mse_array.append((mse_value - base_mse) / number_of_points[exp_to_exclude])

        # Step 2: Perform an exchange by removing an experiment from the worst design point
        # and adding it to the best design point
        argsort_mse = np.argsort(mse_array)
        okay = False
        for m in argsort_mse[:-1]:
            # First non-zero experiment
            worst_experiment = [all_epsilons[arr_id] for arr_id in argsort_mse[::-1]
                                if len(all_epsilons[arr_id]) > 0][0]
            data = random.choice(worst_experiment)
            arr_length = len(worst_experiment)
            worst_experiment.remove(data)
            assert len(worst_experiment) == arr_length - 1

            epsilon = maximum_deviation(number_of_points[m], dimension, estimator, feature_generator)
            all_epsilons[m].append(epsilon)

            # Recalculate mean values and MSE
            mean_values = np.array([np.mean(x) for x in all_epsilons if len(x) > 0])
            number_of_points_tmp = np.array([number_of_points[id] for id, x in enumerate(all_epsilons)
                                             if len(x) > 0])
            assert len(number_of_points_tmp) == len(mean_values)
            h_est = vcd_mse(max_vcd, number_of_points_tmp, mean_values)
            mse_value = mse(h_est, number_of_points_tmp, mean_values)

            # If the MSE decreases, accept the exchange
            if mse_value < base_mse:
                base_mse = mse_value
                okay = True
                break
            else:
                # Reverse the exchange if the MSE increases
                worst_experiment.append(data)
                all_epsilons[m].remove(epsilon)

        # Step 3: Repeat until stopping criteria are met
        if not okay:
            break


def maximum_deviation(n_samples, dimension, estimator, feature_generator):
    # Generate a random dataset
    X = np.random.random(size=(n_samples, dimension))
    y_class = np.random.randint(0, 1, size=(n_samples,))
    if feature_generator is not None:
        X = feature_generator(X)
    # Create two datasets, one with normal labels and another with reversed labels
    Z1 = np.hstack((X, y_class.reshape(-1, 1)))
    Z2 = np.hstack((X, (1 - y_class).reshape(-1, 1)))
    # Split the dataset into two parts
    n1 = int(len(X) / 2)
    Z1_1 = Z1[:n1]
    Z1_2 = Z1[n1:]
    Z2_2 = Z2[n1:]
    # Fit the model on the merged dataset
    estimator.fit(np.vstack((Z1_1[:, :-1], Z2_2[:, :-1])),
                  np.vstack((Z1_1[:, -1].reshape(-1, 1), Z2_2[:, -1].reshape(-1, 1))))
    # Calculate the error rates on the two parts of the original dataset
    E1 = mean_absolute_error(Z1_1[:, -1], estimator.predict(Z1_1[:, :-1]) > 0.5)
    E2 = mean_absolute_error(Z1_2[:, -1], estimator.predict(Z1_2[:, :-1]) > 0.5)
    # Calculate the maximum deviation
    epsilon = abs(E1 - E2)
    return epsilon


if __name__ == '__main__':
    # X, y = load_diabetes(return_X_y=True)
    X, y = make_friedman1()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardize X
    scaler_X = StandardScaler()
    X_train_standardized = scaler_X.fit_transform(X_train)
    X_test_standardized = scaler_X.transform(X_test)

    # Standardize y
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_standardized = (y_train - y_mean) / y_std
    y_test_standardized = (y_test - y_mean) / y_std

    estimator = Ridge(alpha=0.01)
    estimator.fit(X_train, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(X_test)))
    print(vc_dimension_estimation(X, y, estimator))

    estimator = Pipeline(steps=[('PolynomialFeatures', PolynomialFeatures(degree=3)), ('Ridge', Ridge(alpha=0.01))])
    estimator.fit(X_train, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(X_test)))
    print(vc_dimension_estimation(X, y, estimator))
