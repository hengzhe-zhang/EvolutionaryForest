import unittest

import numpy as np
from sklearn.datasets import make_regression

from evolutionary_forest.utility.ood_split import OutOfDistributionSplit


class TestOutOfDistributionSplit(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset
        X, y = make_regression(n_samples=100, n_features=5)
        self.X = X
        self.y = y
        self.cv = OutOfDistributionSplit(n_splits=5, n_bins=5)

    def test_number_of_splits(self):
        # Test if the number of splits is correct
        splits = list(self.cv.split(self.X, self.y))
        self.assertEqual(len(splits), self.cv.get_n_splits(self.X, self.y))

    def test_no_overlap_in_test_sets(self):
        # Test if test sets are mutually exclusive
        splits = list(self.cv.split(self.X, self.y))
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                self.assertFalse(np.intersect1d(splits[i][1], splits[j][1]).size > 0)

    def test_each_split_contains_unique_bin(self):
        # Test if each split contains a unique bin
        binned_y = self.cv._bin_y(self.y)
        splits = list(self.cv.split(self.X, self.y))
        for train_idx, test_idx in splits:
            test_bins = np.unique(binned_y[test_idx])
            self.assertEqual(len(test_bins), 1)

    def test_exception_for_invalid_binning_strategy(self):
        # Test if an exception is raised for an invalid binning strategy
        with self.assertRaises(ValueError):
            invalid_cv = OutOfDistributionSplit(
                n_splits=5, n_bins=5, binning_strategy="invalid"
            )
            list(invalid_cv.split(self.X, self.y))


if __name__ == "__main__":
    unittest.main()
