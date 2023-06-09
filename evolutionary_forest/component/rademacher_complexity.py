import copy

import numpy as np
import scipy
from scipy.stats import mannwhitneyu
from sklearn.datasets import make_friedman1
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

number_samples = 20


def generate_rademacher_vector(X):
    # Generate a random_vector containing Rademacher random variables (+1, -1).
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 2, (number_samples, len(X))) * 2 - 1


def rademacher_complexity_estimation(X, y, estimator, random_rademacher_vector,
                                     reference_complexity_list=None,
                                     objective_weight=0.1):
    """
    Calculates the fitness of a candidate solution/individual by using the relative
    squared errors (RSE) and the Rademacher Complexity.

    :return: individual fitness
    """

    # Relative Squared Error
    r2 = r2_score(y, estimator.predict(X))
    mse = mean_squared_error(y, estimator.predict(X))
    normalize_factor = np.mean((np.mean(y) - y) ** 2)

    estimator = copy.deepcopy(estimator)
    calculate_correlation = lambda sigma, fx: np.sum(sigma * fx) / len(fx)
    complexity = []
    bounded_complexity = []
    for s in range(number_samples):
        # estimator.fit(X, y * random_rademacher_vector[s])
        # normalized_squared_error = (estimator.predict(X) - y) ** 2 / normalize_factor
        # correlations = calculate_correlation(random_rademacher_vector[s], normalized_squared_error)
        # bounded_mse = np.clip(normalized_squared_error, 0, 1)
        # bounded_correlation = calculate_correlation(random_rademacher_vector[s],
        #                                             bounded_mse)
        # complexity.append(np.abs(correlations))
        # bounded_complexity.append(np.abs(bounded_correlation))

        bounded_mse = mse
        try:
            weight = np.abs(scipy.linalg.pinv((np.reshape(random_rademacher_vector[s], (-1, 1)) * X).T @ X) @ \
                            (random_rademacher_vector[s] * X.T) @ np.reshape(y, (-1, 1)))
            rademacher_a = random_rademacher_vector[s].T @ ((weight.T @ X.T).flatten() - y) ** 2
            weight = np.abs(scipy.linalg.pinv((np.reshape(-random_rademacher_vector[s], (-1, 1)) * X).T @ X) @ \
                            (-random_rademacher_vector[s] * X.T) @ np.reshape(y, (-1, 1)))
            rademacher_b = -random_rademacher_vector[s].T @ ((weight.T @ X.T).flatten() - y) ** 2
            rademacher = max(rademacher_a, rademacher_b)
        except:
            rademacher = np.inf
        complexity.append(rademacher)
        bounded_complexity.append(rademacher)
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
    # print(rademacher_complexity)
    return ((r2, 1), (rademacher_complexity, -objective_weight)), bounded_rademacher_complexity, bounded_complexity


if __name__ == '__main__':
    # X, y = load_diabetes(return_X_y=True)
    X, y = make_friedman1(n_samples=50)

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

    estimator = Ridge(alpha=0.1)
    estimator.fit(X_train, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(X_test)))
    print(rademacher_complexity_estimation(X_train, y_train, estimator, random_rademacher_vector))

    estimator = Pipeline(steps=[('PolynomialFeatures', PolynomialFeatures(degree=2)), ('Ridge', Ridge(alpha=0.1))])
    estimator.fit(X_train, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(X_test)))
    print(rademacher_complexity_estimation(X_train, y_train, estimator, random_rademacher_vector))

    estimator = Pipeline(steps=[('PolynomialFeatures', PolynomialFeatures(degree=3)), ('Ridge', Ridge(alpha=0.1))])
    estimator.fit(X_train, y_train)
    # Calculate the R2 score on the test set
    print('Test R2', r2_score(y_test, estimator.predict(X_test)))
    print(rademacher_complexity_estimation(X_train, y_train, estimator, random_rademacher_vector))
