from numba import njit
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class StringToIntMapper(TransformerMixin):
    def __init__(self):
        self.mapping = {}
        self.next_index = 0  # Tracks the next available integer index

    def fit(self, X):
        # Ensure X is a 1D numpy array of strings
        for s in np.unique(X):
            self._update_mapping_and_transform(s)

    def transform(self, X):
        # Ensure X is a 1D numpy array of strings
        for s in np.unique(X):
            self._update_mapping_and_transform(s)
        return np.vectorize(self.mapping.get)(X)

    def _update_mapping_and_transform(self, string):
        # Check if string is already in mapping, if not, add it
        if string not in self.mapping:
            self.mapping[string] = self.next_index
            self.next_index += 1
        return self.mapping[string]


class GroupByAggregator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_keys = None
        self.aggregated_values = None
        self.mapper = StringToIntMapper()

    # @staticmethod
    # def _aggregator(keys, values, unique_keys):
    #     pass

    def fit(self, X, y=None):
        assert X.shape[1] == 2, "X must have 2 columns"
        # self.unique_keys = np.unique(X[:, 0])
        self.mapper.fit(X[:, 0])
        self.aggregated_values = self._aggregator(
            self.mapper.transform(X[:, 0]),
            X[:, 1].astype(np.float32),
            np.unique(self.mapper.transform(X[:, 0])),
        )
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
            self.mapper.transform(X[:, 0]),
            X[:, 1].astype(np.float32),
            np.unique(self.mapper.transform(X[:, 0])),
            self.aggregated_values,
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


class GroupByCountTransformer(GroupByAggregator):
    @staticmethod
    @njit(cache=True)
    def _aggregator(keys, values, unique_keys):
        count_values = np.zeros_like(unique_keys, dtype=np.float64)

        for i in range(len(unique_keys)):
            count_values[i] = np.sum(keys == unique_keys[i])

        return count_values


if __name__ == "__main__":
    # Create some example data
    data = np.array([["1", 5], [2, 6], [1, 5], [3, 7], [2, 8], [1, 5]])

    # Instantiate the transformer
    transformer = GroupByMeanTransformer()

    # Fit and transform the data
    transformer.fit(data)
    result = transformer.transform(data)
    print(result)
