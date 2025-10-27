import numpy as np
import faiss
from sklearn.base import BaseEstimator, RegressorMixin


class FaissKNNRegressor(BaseEstimator, RegressorMixin):
    """Simplest FAISS-based KNN Regressor (sklearn-style)."""

    def __init__(self, n_neighbors=5, metric="l2", n_threads=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_threads = n_threads

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._y = y
        d = X.shape[1]

        faiss.omp_set_num_threads(self.n_threads)

        if self.metric == "l2":
            self.index_ = faiss.IndexFlatL2(d)
        elif self.metric == "ip":
            self.index_ = faiss.IndexFlatIP(d)
        else:
            raise ValueError("metric must be 'l2' or 'ip'")

        self.index_.add(X)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        _, indices = self.index_.search(X, self.n_neighbors)
        return np.mean(self._y[indices], axis=1)
