from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np


class OneHotStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-8):
        self.scalers_ = {}
        self.eps = eps  # Small epsilon value to account for floating-point errors

    def fit(self, X, y=None):
        # Iterate over each column in X
        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            # Check if the column only contains values close to 0 and 1
            if len(unique_vals) == 2 and np.allclose(
                unique_vals, [0, 1], atol=self.eps
            ):
                # If only 0 and 1 are present, no scaling will be applied
                self.scalers_[i] = None
            else:
                # Apply standard scaling
                scaler = StandardScaler()
                scaler.fit(X[:, i].reshape(-1, 1))
                self.scalers_[i] = scaler
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for i, scaler in self.scalers_.items():
            if scaler is not None:
                X_transformed[:, i] = scaler.transform(X[:, i].reshape(-1, 1)).flatten()
        return X_transformed


if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = np.array(
        [[1, 200, 0], [0, 150, 1], [1, 300, 0], [0, 250, 1], [1, 100, 0]], dtype=float
    )

    # Initialize and fit the custom scaler
    scaler = OneHotStandardScaler()
    scaler.fit(X)

    # Transform the data
    X_scaled = scaler.transform(X)

    print(X_scaled)
