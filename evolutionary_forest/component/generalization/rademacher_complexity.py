import copy

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.datasets import make_friedman1
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from evolutionary_forest.component.pac_bayesian import PACBayesianConfiguration

number_samples = 20


def generate_rademacher_vector(X):
    # Generate a random_vector containing Rademacher random variables (+1, -1).
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 2, (number_samples, len(X))) * 2 - 1


def rademacher_complexity_estimation(X, y, estimator, random_rademacher_vector,
                                     reference_complexity_list=None,
                                     configuration: PACBayesianConfiguration = None,
                                     rademacher_mode='Analytical'):
    """
    Calculates the fitness of a candidate solution/individual by using the relative
    squared errors (RSE) and the Rademacher Complexity.

    :return: individual fitness
    """
    objective_weight = 1

    # Relative Squared Error
    r2 = r2_score(y, estimator.predict(X))
    normalize_factor = np.mean((np.mean(y) - y) ** 2)

    estimator = copy.deepcopy(estimator)
    calculate_correlation = lambda sigma, fx: np.sum(sigma * fx) / len(fx)
    complexity = []
    bounded_complexity = []
    for s in range(number_samples):
        # if Rademacher is 1, then try to fit -y
        # if Rademacher is -1, then try to fit y
        if rademacher_mode == 'Local':
            rademacher_target = -y * random_rademacher_vector[s]
            estimator.fit(np.concatenate([X, X], axis=0),
                          np.concatenate([y, rademacher_target], axis=0))
        elif rademacher_mode == 'Global':
            rademacher_target = -y * random_rademacher_vector[s]
            estimator.fit(X, rademacher_target)
        else:
            raise Exception

        # maximize MSE when Rademacher is 1
        normalized_squared_error = (estimator.predict(X) - y) ** 2 / normalize_factor
        rademacher = calculate_correlation(random_rademacher_vector[s], normalized_squared_error)
        rademacher = max(rademacher, 0)
        complexity.append(rademacher)

        if configuration.bound_reduction:
            bounded_mse = np.clip(normalized_squared_error, 0, 1)
        else:
            bounded_mse = normalized_squared_error
        bounded_rademacher = calculate_correlation(random_rademacher_vector[s], bounded_mse)
        bounded_rademacher = max(bounded_rademacher, 0)
        bounded_complexity.append(bounded_rademacher)
        if reference_complexity_list is not None:
            mannwhitneyu_result = mannwhitneyu(reference_complexity_list,
                                               np.mean(bounded_mse) + 2 * np.array(bounded_complexity),
                                               alternative='less')
            if mannwhitneyu_result.pvalue < 0.05:
                # complexity is significantly larger than the reference complexity
                break
    # Calculate the rademacher complexity as the average of the complexity values.
    rademacher_complexity = np.mean(complexity)
    bounded_rademacher_complexity = np.mean(bounded_complexity)
    # Maximize R2, Minimize Rademacher Complexity
    return ((r2, 1), (rademacher_complexity, -objective_weight)), bounded_rademacher_complexity, bounded_complexity


if __name__ == '__main__':
    # X, y = load_diabetes(return_X_y=True)
    X, y = make_friedman1(n_samples=50, n_features=10)

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

    random_rademacher_vector = generate_rademacher_vector(X_train)
    print(random_rademacher_vector)
    random_rademacher_vector_b = generate_rademacher_vector(X_train)
    print(random_rademacher_vector_b)
    if np.array_equal(random_rademacher_vector, random_rademacher_vector_b):
        print("The two arrays are identical.")
    else:
        print("The two arrays are different.")

    for rademacher_mode in ['Local', 'LeastSquare']:
        estimator = Ridge(alpha=0.1)
        # estimator = LinearRegression()
        estimator.fit(X_train, y_train)
        # Calculate the R2 score on the test set
        print('Test R2', r2_score(y_test, estimator.predict(X_test)))
        print(rademacher_complexity_estimation(X_train, y_train, estimator, random_rademacher_vector,
                                               rademacher_mode=rademacher_mode))

        poly = PolynomialFeatures(degree=2)
        estimator.fit(poly.fit_transform(X_train), y_train)
        # Calculate the R2 score on the test set
        print('Test R2', r2_score(y_test, estimator.predict(poly.transform(X_test))))
        print(rademacher_complexity_estimation(poly.transform(X_train),
                                               y_train, estimator, random_rademacher_vector,
                                               rademacher_mode=rademacher_mode))

        poly = PolynomialFeatures(degree=3)
        estimator.fit(poly.fit_transform(X_train), y_train)
        # Calculate the R2 score on the test set
        print('Test R2', r2_score(y_test, estimator.predict(poly.transform(X_test))))
        print(rademacher_complexity_estimation(poly.fit_transform(X_train),
                                               y_train, estimator, random_rademacher_vector,
                                               rademacher_mode=rademacher_mode))
