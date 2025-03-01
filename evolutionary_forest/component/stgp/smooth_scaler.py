import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NearestValueTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_values_ = None

    def fit(self, X, y=None):
        # Flatten the input and sort the unique values once
        X = np.array(X).flatten()
        self.unique_values_ = np.unique(X)
        return self

    def transform(self, X):
        if self.unique_values_ is None:
            raise ValueError("The model has not been fitted yet!")

        X = np.array(X).flatten()
        # np.searchsorted finds indices where elements should be inserted to maintain order
        indices = np.searchsorted(self.unique_values_, X, side="left")

        # Handle edge cases for boundary values
        indices = np.clip(indices, 1, len(self.unique_values_) - 1)
        left = self.unique_values_[indices - 1]
        right = self.unique_values_[indices]

        # Select the nearest value among the candidates (left or right)
        nearest_values = np.where((X - left) <= (right - X), left, right)
        return nearest_values


class NearestValueTransformer2D(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformers_ = None

    def fit(self, X, y=None):
        num_columns = X.shape[1]  # Number of columns in X

        # Fit a transformer for each column
        self.transformers_ = []
        for i in range(num_columns):
            transformer = NearestValueTransformer()
            transformer.fit(X[:, i])  # Fit transformer to column i
            self.transformers_.append(transformer)
        return self

    def transform(self, X):
        num_columns = X.shape[1]
        transformed_columns = []

        # Transform each column using the corresponding fitted transformer
        for i in range(num_columns):
            transformed_column = self.transformers_[i].transform(X[:, i])
            transformed_columns.append(transformed_column)

        # Stack transformed columns back into a 2D array
        return np.column_stack(transformed_columns)


if __name__ == "__main__":
    # Example usage:
    data = np.random.rand(100) * 100  # Random data
    transformer = NearestValueTransformer()
    transformer.fit(data)

    # Transform the data by adding small noise to see the effect
    without_transform = data + np.random.normal(0, 1, size=data.shape)
    print("Transformed Data:", without_transform[:5])
    transformed_data = transformer.transform(without_transform)
    print("Original Data:", data[:5])
    print("Original Data:", transformer.transform(data[:5]))
    print("Transformed Data:", transformed_data[:5])
