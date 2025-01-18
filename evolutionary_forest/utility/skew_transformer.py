import numpy as np
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin


class SkewnessCorrector(BaseEstimator, TransformerMixin):
    def __init__(self, skewness_mode="both"):
        # Add a parameter to specify which skewness to handle ('left', 'right', 'both')
        self.skewness_mode = skewness_mode

    def fit(self, X, y=None):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        self.skewness_ = skew(X, axis=0)
        return self

    def transform(self, X, y=None):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        X_trans = np.empty_like(X)

        for i in range(X.shape[1]):
            signs = np.sign(X[:, i])
            abs_values = np.abs(X[:, i])

            if self.skewness_[i] > 0.5 and self.skewness_mode in ["right", "both"]:
                transformed = np.sqrt(abs_values)
            elif self.skewness_[i] < -0.5 and self.skewness_mode in ["left", "both"]:
                transformed = abs_values**2
            else:
                transformed = abs_values

            X_trans[:, i] = transformed * signs

        return X_trans.flatten() if X_trans.shape[1] == 1 else X_trans

    def inverse_transform(self, X, y=None):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        X_inv_trans = np.empty_like(X)

        for i in range(X.shape[1]):
            signs = np.sign(X[:, i])
            abs_values = np.abs(X[:, i])

            if self.skewness_[i] > 0.5 and self.skewness_mode in ["right", "both"]:
                inv_transformed = abs_values**2
            elif self.skewness_[i] < -0.5 and self.skewness_mode in ["left", "both"]:
                inv_transformed = np.sqrt(abs_values)
            else:
                inv_transformed = abs_values

            X_inv_trans[:, i] = inv_transformed * signs

        return X_inv_trans.flatten() if X_inv_trans.shape[1] == 1 else X_inv_trans


class CubeSkewnessCorrector(SkewnessCorrector):
    def transform(self, X, y=None):
        X_trans = X.reshape(-1, 1) if X.ndim == 1 else X

        for i in range(X_trans.shape[1]):
            if self.skewness_[i] > 1 and self.skewness_mode in [
                "right",
                "both",
            ]:  # More severe skewness
                X_trans[:, i] = np.cbrt(X_trans[:, i])  # Apply cbrt for right skew
            elif self.skewness_[i] < -1 and self.skewness_mode in ["left", "both"]:
                X_trans[:, i] = (
                    X[:, i] ** 3
                )  # Apply cube directly without abs for left skew

        return X_trans

    def inverse_transform(self, X, y=None):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        X_inv_trans = np.empty_like(X)

        for i in range(X.shape[1]):
            if self.skewness_[i] > 1 and self.skewness_mode in ["right", "both"]:
                X_inv_trans[:, i] = np.sign(X[:, i]) * (
                    np.abs(X[:, i]) ** 3
                )  # Inverse of cbrt
            elif self.skewness_[i] < -1 and self.skewness_mode in ["left", "both"]:
                X_inv_trans[:, i] = np.cbrt(
                    X[:, i]
                )  # Inverse of cube is cbrt, applied directly
            else:
                X_inv_trans[:, i] = (
                    super().inverse_transform(X[:, i].reshape(-1, 1)).flatten()
                )  # Use base class inverse

        return X_inv_trans.flatten() if X_inv_trans.shape[1] == 1 else X_inv_trans


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    y = y**3
    print(y[:5])
    print("Skew", skew(y, axis=0))
    sc = CubeSkewnessCorrector()
    y_transformed = sc.fit_transform(y)
    print("Skew", skew(y_transformed, axis=0))
    print(sc.inverse_transform(y_transformed)[:5])
