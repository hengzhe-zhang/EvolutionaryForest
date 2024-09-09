from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class YIntScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler
        self.is_integer = False
        self.is_positive = False
        self.is_negative = False

    def fit(self, y):
        # Check if the original data was integer and if the rounded value is still integer
        if np.all(np.equal(y, np.round(y))):
            self.is_integer = True
        if np.all(y >= 0):
            self.is_positive = True
        if np.all(y <= 0):
            self.is_negative = True
        # Fit the underlying scaler
        self.scaler.fit(y)
        return self

    def transform(self, y):
        return self.scaler.transform(y)

    def inverse_transform(self, y):
        # Inverse transform using the underlying scaler
        y_transformed = self.scaler.inverse_transform(y)
        # If the original data was integer, round the transformed data
        if self.is_integer:
            y_transformed = np.round(y_transformed)
        if self.is_positive:
            y_transformed = np.clip(y_transformed, 0, None)
        if self.is_negative:
            y_transformed = np.clip(y_transformed, None, 0)
        return y_transformed


if __name__ == "__main__":
    # Example usage:
    from sklearn.preprocessing import StandardScaler

    # Instantiate the base scaler (e.g., StandardScaler)
    base_scaler = StandardScaler()

    # Wrap the base scaler with the YScaler
    y_scaler = YIntScaler(scaler=base_scaler)

    # Example data
    y = np.array([[1], [1], [2], [2], [3]])

    # Fit the wrapper scaler
    y_scaler.fit(y)

    # Transform and inverse transform example
    y_transformed = y_scaler.transform(y)
    y_inverse_transformed = y_scaler.inverse_transform(
        y_transformed + np.random.normal(0, 0.1, y.shape)
    )

    print("Original data:\n", y)
    print("Transformed data:\n", y_transformed)
    print("Inverse transformed data:\n", y_inverse_transformed)
