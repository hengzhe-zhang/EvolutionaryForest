import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin


class SkewnessCorrector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Reshape X to 2D if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Compute skewness along each feature
        self.skewness_ = skew(X, axis=0)
        # Compute minimum value for each feature to apply shift if necessary
        self.min_ = X.min(axis=0)
        return self

    def transform(self, X, y=None):
        # Reshape X to 2D if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        original_shape = X.shape

        for i in range(X.shape[1]):
            shift = 0
            if self.min_[i] < 0:
                shift = -self.min_[i]  # Shift to make the minimum value 0

            if self.skewness_[i] > 0.5:
                # Right-skewed, apply square root after shifting
                X[:, i] = np.sqrt(X[:, i] + shift)
                # Reverse the shift effect by subtracting the square root of the shift
                X[:, i] = X[:, i] - np.sqrt(shift)
            elif self.skewness_[i] < -0.5:
                # Left-skewed, apply square after shifting
                X[:, i] = (X[:, i] + shift) ** 2
                # Reverse the shift effect by subtracting the squared shift (considering the data was squared)
                X[:, i] = X[:, i] - shift**2

        # If the original input was 1D, return a 1D array
        if original_shape[0] == 1 or original_shape[1] == 1:
            return X.flatten()
        else:
            return X

    def inverse_transform(self, X, y=None):
        # Reshape X to 2D if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        original_shape = X.shape

        for i in range(X.shape[1]):
            shift = 0
            if self.min_[i] < 0:
                shift = -self.min_[i]  # Original shift to make the minimum value 0

            if self.skewness_[i] > 0.5:
                # Data was right-skewed and underwent a square root transformation
                # Add the square root of the shift, then square to reverse the transformation
                X[:, i] = (X[:, i] + np.sqrt(shift)) ** 2
                # Reverse the original shift
                X[:, i] = X[:, i] - shift
            elif self.skewness_[i] < -0.5:
                # Data was left-skewed and underwent a square transformation
                # Add the squared shift, then take the square root to reverse the transformation
                X[:, i] = np.sqrt(X[:, i] + shift**2)
                # Reverse the original shift
                X[:, i] = X[:, i] - shift

        # If the original input was 1D, return a 1D array
        if original_shape[0] == 1 or original_shape[1] == 1:
            return X.flatten()
        else:
            return X


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression, load_diabetes

    X, y = load_diabetes(return_X_y=True)
    y = y**3
    print(y[:5])
    print("Skew", skew(y, axis=0))
    sc = SkewnessCorrector()
    y_transformed = sc.fit_transform(y)
    print("Skew", skew(y_transformed, axis=0))
    print(sc.inverse_transform(y_transformed)[:5])
