import copy
import math

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


def vc_dimension_estimation(X, y, estimator):
    Yp = estimator.predict(X)
    rse = 1 - r2_score(y, Yp)

    estimator = copy.deepcopy(estimator)
    number_of_points = [10, 20, 30, 50, 100, 200, 500]
    all_epsilons = []

    for n_samples in number_of_points:
        epsilons = []
        # Number of trials
        m = 20
        for _ in range(m):
            # Generate a random dataset
            X = np.random.random(size=(n_samples, 10))
            y_class = np.random.randint(0, 1, size=(n_samples,))

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
            epsilons.append(epsilon)

        # Calculate the average maximum deviation
        epsilons = np.mean(epsilons)
        all_epsilons.append(epsilons)

    # Calculate VC dimension
    hs = np.arange(1, 1000)

    def mse(h):
        all_mse = []
        for n_samples, epsilons in zip(number_of_points, all_epsilons):
            en = (n_samples / 2) / h
            all_mse.append((phi(en) - epsilons) ** 2)
        return np.mean(all_mse)

    h_est = hs[np.argmin(list(map(mse, hs)))]
    # return exp_err(math.sqrt(mean_squared_error(y, Yp)), h_est, len(X))
    return ((rse, -1), (h_est, -0.1))


if __name__ == '__main__':
    X, y = load_diabetes(return_X_y=True)

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
