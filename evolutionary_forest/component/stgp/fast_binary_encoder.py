import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin
from numba import njit


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown
        self.category_mappings = None
        self.unknown_code = None
        self.n_bits = None

    def fit(self, X, y=None):
        # Assuming X is a 1D numpy array of categorical features
        unique_categories = np.unique(X)
        n_categories = len(unique_categories)
        self.n_bits = math.ceil(
            math.log2(n_categories + 1)
        )  # +1 for a unique "unknown" category

        # Map categories to their integer indices
        self.category_mappings = {
            category: idx for idx, category in enumerate(unique_categories)
        }

        # Define the integer code for unknown values
        self.unknown_code = -1 if self.handle_unknown == "ignore" else n_categories

        return self

    def transform(self, X):
        # Map categories to integers and then convert to binary
        transformed = np.array(
            [self.category_mappings.get(x, self.unknown_code) for x in X]
        )
        return self._to_binary(transformed)

    def _to_binary(self, X):
        # Efficient binary conversion using Numba
        return to_binary_numba(X, self.n_bits)


@njit(cache=True)
def to_binary_numba(X, n_bits):
    # Allocate an array for the binary representation
    rows = len(X)
    binary_array = np.zeros((rows, n_bits), dtype=np.uint8)

    for i in range(rows):
        # Convert each integer to binary and fill the array
        num = X[i]
        for bit in range(n_bits):
            binary_array[i, n_bits - 1 - bit] = num % 2
            num //= 2

    return binary_array


if __name__ == "__main__":
    # Example usage
    X = np.array(["cat", "dog", "bird", "dog", "cat", "fish", "A", "B", "C", "D", "E"])
    encoder = BinaryEncoder()
    encoder.fit(X)
    encoded_X = encoder.transform(X)
    print(encoded_X)
    print(encoder.transform(["penguin"]))
