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
    @njit(cache=True)
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
    @njit(cache=True)
    def _aggregator(keys, values, unique_keys):
        means = np.zeros(len(unique_keys), dtype=np.float64)

        for i, key in enumerate(unique_keys):
            # Extract values corresponding to the current unique key
            group_values = values[keys == key]
            # Compute the mean of the group values directly
            means[i] = np.mean(group_values)

        return means


class GroupByMedianTransformer(GroupByAggregator):
    @staticmethod
    @njit(cache=True)
    def _aggregator(keys, values, unique_keys):
        median_values = np.zeros_like(unique_keys, dtype=np.float64)

        for i in range(len(unique_keys)):
            group_values = values[keys == unique_keys[i]]
            median_values[i] = np.median(group_values)

        return median_values


class GroupByMinTransformer(GroupByAggregator):
    @staticmethod
    @njit(cache=True)
    def _aggregator(keys, values, unique_keys):
        min_values = np.zeros_like(unique_keys, dtype=np.float64)

        for i in range(len(unique_keys)):
            group_values = values[keys == unique_keys[i]]
            min_values[i] = np.min(group_values)

        return min_values


class GroupByMaxTransformer(GroupByAggregator):
    @staticmethod
    @njit(cache=True)
    def _aggregator(keys, values, unique_keys):
        max_values = np.zeros_like(unique_keys, dtype=np.float64)

        for i in range(len(unique_keys)):
            group_values = values[keys == unique_keys[i]]
            max_values[i] = np.max(group_values)

        return max_values
