"""Categorical encoder utilities."""

import numpy as np
import pandas as pd
import sklearn
from category_encoders import TargetEncoder as SimpleTargetEncoder
from packaging import version
from sklearn.preprocessing import OneHotEncoder


class OneHotTargetEncoder:
    """OneHot first, then target-encode those one-hot features (chain)."""

    def __init__(self, cols=None):
        self.cols = cols
        sparse_kw = {"sparse": False} if version.parse(sklearn.__version__) < version.parse("1.2") else {
            "sparse_output": False}
        self._onehot = OneHotEncoder(handle_unknown="ignore", drop="if_binary", **sparse_kw)
        self._target = SimpleTargetEncoder()

    def fit(self, X, y=None):
        X_cat = np.asarray(X)[:, self.cols] if self.cols is not None else np.asarray(X)
        self._onehot.fit(X_cat)
        A = self._onehot.transform(X_cat)
        self._target.fit(pd.DataFrame(A).astype(str), y)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_cat = X[:, self.cols] if self.cols is not None else X
        A = self._onehot.transform(X_cat)
        enc = np.asarray(self._target.transform(pd.DataFrame(A).astype(str)))
        if self.cols is None:
            return enc
        noncat = [i for i in range(X.shape[1]) if i not in self.cols]
        return np.hstack([X[:, noncat], enc])

    def fit_transform(self, X, y=None):
        X = np.asarray(X)
        X_cat = X[:, self.cols] if self.cols is not None else X
        A = self._onehot.fit_transform(X_cat)
        enc = np.asarray(self._target.fit_transform(pd.DataFrame(A).astype(str), y))
        if self.cols is None:
            return enc
        noncat = [i for i in range(X.shape[1]) if i not in self.cols]
        return np.hstack([X[:, noncat], enc])


if __name__ == "__main__":
    # Categorical-only input (cols=None)
    # Multi-class: 3 values in col0, 3 in col1
    X_cat = np.array([["a", "x"], ["b", "y"], ["c", "z"], ["a", "y"], ["b", "z"], ["c", "x"]])
    y = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 1.8])
    enc = OneHotTargetEncoder()
    out = np.asarray(enc.fit_transform(X_cat, y), dtype=float)
    print("cols=None (categorical only):")
    print("X shape:", X_cat.shape, "-> out shape:", out.shape)
    print(out)

    # Full X with cols (categorical at indices 1, 3)
    X = np.column_stack([[1, 2, 3, 4, 5, 6], X_cat[:, 0], [10, 20, 30, 40, 50, 60], X_cat[:, 1]])
    enc2 = OneHotTargetEncoder(cols=[1, 3])
    out2 = np.asarray(enc2.fit_transform(X, y), dtype=float)
    print("cols=[1,3] (full X, cat at 1,3):")
    print("X shape:", X.shape, "-> out shape:", out2.shape)
    print(out2)
