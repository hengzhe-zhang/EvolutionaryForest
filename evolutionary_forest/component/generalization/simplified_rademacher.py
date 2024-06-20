import copy
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

number_samples = 20


def generate_rademacher_vector(X):
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 2, (number_samples, len(X))) * 2 - 1


def rademacher_complexity_estimation(
    X,
    y,
    estimator,
    random_rademacher_vector,
):
    objective_weight = 1
    r2 = r2_score(y, estimator.predict(X))
    normalize_factor = np.mean((np.mean(y) - y) ** 2)

    estimator = copy.deepcopy(estimator)
    calculate_correlation = lambda sigma, fx: np.sum(sigma * fx) / len(fx)
    complexity = []
    for s in range(number_samples):
        rademacher_target = -y * random_rademacher_vector[s]
        estimator.fit(
            np.concatenate([X, X], axis=0),
            np.concatenate([y, rademacher_target], axis=0),
        )

        normalized_squared_error = (estimator.predict(X) - y) ** 2 / normalize_factor
        rademacher = calculate_correlation(
            random_rademacher_vector[s], normalized_squared_error
        )
        rademacher = max(rademacher, 0)
        complexity.append(rademacher)

    rademacher_complexity = np.mean(complexity)
    return (((r2, 1), (rademacher_complexity, -objective_weight)),)


if __name__ == "__main__":
    X, y = make_friedman1(n_samples=50, n_features=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler_X = StandardScaler()
    X_train_standardized = scaler_X.fit_transform(X_train)
    X_test_standardized = scaler_X.transform(X_test)

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_standardized = (y_train - y_mean) / y_std
    y_test_standardized = (y_test - y_mean) / y_std

    random_rademacher_vector = generate_rademacher_vector(X_train)
    print(random_rademacher_vector)

    estimator = Ridge(alpha=0.1)
    estimator.fit(X_train, y_train)
    print("Test R2", r2_score(y_test, estimator.predict(X_test)))
    print(
        rademacher_complexity_estimation(
            X_train,
            y_train,
            estimator,
            random_rademacher_vector,
        )
    )

    poly = PolynomialFeatures(degree=2)
    estimator.fit(poly.fit_transform(X_train), y_train)
    print("Test R2", r2_score(y_test, estimator.predict(poly.transform(X_test))))
    print(
        rademacher_complexity_estimation(
            poly.transform(X_train),
            y_train,
            estimator,
            random_rademacher_vector,
        )
    )

    poly = PolynomialFeatures(degree=3)
    estimator.fit(poly.fit_transform(X_train), y_train)
    print("Test R2", r2_score(y_test, estimator.predict(poly.transform(X_test))))
    print(
        rademacher_complexity_estimation(
            poly.fit_transform(X_train),
            y_train,
            estimator,
            random_rademacher_vector,
        )
    )
