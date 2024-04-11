import numpy as np
from numba import njit
from sklearn.base import TransformerMixin, BaseEstimator


class GroupByMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_keys = None
        self.means = None

    @njit
    def _groupby_mean_fit(self, keys, values):
        unique_keys = np.unique(keys)
        means = np.zeros_like(unique_keys, dtype=np.float64)
        counts = np.zeros_like(unique_keys, dtype=np.int64)

        # Calculate the sum and counts for each group
        for i in range(len(keys)):
            # get group id
            idx = np.searchsorted(unique_keys, keys[i])
            means[idx] += values[i]
            counts[idx] += 1

        # Divide the sum by counts to get the mean
        means /= counts

        return unique_keys, means

    @njit
    def _groupby_mean_transform(self, keys, values, unique_keys, means):
        output_array = np.zeros_like(values, dtype=np.float64)
        for i in range(len(keys)):
            # get group id
            idx = np.searchsorted(unique_keys, keys[i])
            output_array[i] = means[idx]

        return output_array

    def fit(self, X, y=None):
        assert X.shape[1] == 2, "X must have 2 columns"
        self.unique_keys, self.means = self._groupby_mean_fit(X[:, 0], X[:, 1])
        return self

    def transform(self, X):
        assert X.shape[1] == 2, "X must have 2 columns"
        transformed_y = self._groupby_mean_transform(
            X[:, 0], X[:, 1], self.unique_keys, self.means
        )
        return transformed_y
