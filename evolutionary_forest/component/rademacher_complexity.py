import copy

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

number_samples = 20


def generate_rademacher_vector(X):
    # Generate a random_vector containing Rademacher random variables (+1, -1).
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 2, (number_samples, len(X))) * 2 - 1


def rademacher_complexity_estimation(X, y, estimator, random_rademacher_vector,
                                     objective_weight=0.1):
    """
    Calculates the fitness of a candidate solution/individual by using the relative
    squared errors (RSE) and the Rademacher Complexity.

    :return: individual fitness
    """

    # Relative Squared Error
    rse = r2_score(y, estimator.predict(X))

    estimator = copy.deepcopy(estimator)
    calculate_correlation = lambda sigma, fx: np.sum(sigma * fx) / len(fx)
    complexity = []
    for s in range(number_samples):
        estimator.fit(X, random_rademacher_vector[s])
        correlations = calculate_correlation(random_rademacher_vector[s], estimator.predict(X))
        complexity.append(correlations)
    # Calculate the rademacher complexity as the average of the complexity values.
    rademacher_complexity = np.mean(complexity)

    return ((rse, 1), (rademacher_complexity, -objective_weight))


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
