import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class OutOfDistributionSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, n_bins=5, binning_strategy="quantile"):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy

    def _bin_y(self, y):
        if self.binning_strategy == "quantile":
            # Bin `y` into quantiles
            return pd.qcut(y, q=self.n_bins, labels=False, duplicates="drop")
        elif self.binning_strategy == "uniform":
            # Bin `y` into uniform intervals
            return pd.cut(y, bins=self.n_bins, labels=False)
        else:
            # You can add more binning strategies as needed
            raise ValueError("Unsupported binning strategy")

    def split(self, X, y, groups=None):
        binned_y = self._bin_y(y)
        unique_bins = np.unique(binned_y)
        for i in range(min(self.n_splits, len(unique_bins))):
            test_mask = binned_y == unique_bins[i]
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(~test_mask)[0]
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return min(self.n_splits, len(np.unique(self._bin_y(y))))


if __name__ == "__main__":
    pass
