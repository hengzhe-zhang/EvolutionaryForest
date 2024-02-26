import numpy as np
from sklearn.preprocessing import StandardScaler


class StandardScaler1D2D(StandardScaler):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, **kwargs):
        # Reshape X to (-1, 1) if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return super().fit(X, y)

    def transform(self, X, copy=None):
        # Reshape X to (-1, 1) if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return super().transform(X, copy)

    def fit_transform(self, X, y=None, **fit_params):
        # Reshape X to (-1, 1) if it's 1D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return super().fit_transform(X, y, **fit_params)


if __name__ == "__main__":
    data = np.array([1, 2, 3, 4, 5])
    custom_scaler = StandardScaler1D2D()
    scaled_data = custom_scaler.fit_transform(data)
    print(scaled_data)
