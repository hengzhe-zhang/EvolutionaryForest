import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class LinearScalingRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)

        self.n_features_in_ = 1
        x_mean, y_mean = np.mean(X), np.mean(y)
        var_x = np.var(X)
        cov_xy = np.mean((X - x_mean) * (y - y_mean))

        self.b_ = 0 if var_x == 0 else cov_xy / var_x
        self.a_ = y_mean - self.b_ * x_mean
        return self

    def predict(self, X):
        X = np.asarray(X).reshape(-1)
        return self.a_ + self.b_ * X


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    # Generate toy dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Use the Linear Scaling Regressor
    lsr = LinearScalingRegressor()
    lsr.fit(X_train, y_train)
    preds = lsr.predict(X_test)
    r2 = lsr.score(X_test, y_test)
    print(f"R^2 score: {r2}")
