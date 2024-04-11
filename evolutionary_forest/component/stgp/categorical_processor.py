import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, TransformerMixin


class GroupByAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_keys = None
        self.aggregated_values = None

    # @staticmethod
    # def _aggregator(keys, values, unique_keys):
    #     pass

    def fit(self, X, y=None):
        assert X.shape[1] == 2, "X must have 2 columns"
        self.unique_keys = np.unique(X[:, 0])
        self.aggregated_values = self._aggregator(X[:, 0], X[:, 1], self.unique_keys)
        return self

    @staticmethod
    # @njit
    def _transform(keys, values, unique_keys, aggregated_values):
        output_array = np.zeros_like(values, dtype=np.float64)
        for i in range(len(keys)):
            idx = np.searchsorted(unique_keys, keys[i])
            output_array[i] = aggregated_values[idx]
        return output_array

    def transform(self, X):
        assert X.shape[1] == 2, "X must have 2 columns"
        transformed_y = self._transform(
            X[:, 0], X[:, 1], self.unique_keys, self.aggregated_values
        )
        return transformed_y


class GroupByMeanTransformer(GroupByAggregator):
    @staticmethod
    # @njit
    def _aggregator(keys, values, unique_keys):
        means = np.zeros_like(unique_keys, dtype=np.float64)
        counts = np.zeros_like(unique_keys, dtype=np.int64)

        for i in range(len(keys)):
            idx = np.searchsorted(unique_keys, keys[i])
            means[idx] += values[i]
            counts[idx] += 1

        means /= counts
        return means
