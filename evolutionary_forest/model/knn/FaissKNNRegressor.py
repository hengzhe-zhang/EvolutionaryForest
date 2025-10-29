import numpy as np
import faiss
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.metrics import r2_score


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


class RobustFaissKNNRegressor(FaissKNNRegressor):
    """FAISS-based KNN Regressor with leave-one-out R² and constant fallback."""

    def __init__(self, n_neighbors=5, metric="l2", n_threads=1, verbose=True):
        super().__init__(n_neighbors=n_neighbors, metric=metric, n_threads=n_threads)
        self.verbose = verbose

    def fit(self, X, y):
        super().fit(X, y)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Query for n_neighbors + 1 to include self
        _, indices = self.index_.search(X, self.n_neighbors + 1)

        # The first neighbor for each sample is itself (distance = 0)
        # So we remove it in a vectorized way
        indices = indices[:, 1 : self.n_neighbors + 1]

        # Compute mean prediction per sample (vectorized)
        y_pred_train = np.mean(self._y[indices], axis=1)

        # Compute leave-one-out R²
        self.train_score_ = r2_score(y, y_pred_train)

        if self.verbose:
            print(f"Training leave-one-out R² score: {self.train_score_:.4f}")

        # Fallback if score ≤ 0
        if self.train_score_ <= 0:
            self.constant_ = np.mean(y)
            self.use_constant_ = True
            if self.verbose:
                print(
                    f"⚠️ Poor performance — using constant fallback: {self.constant_:.4f}"
                )
        else:
            self.use_constant_ = False

        return self

    def predict(self, X):
        if getattr(self, "use_constant_", False):
            return np.full((len(X),), self.constant_, dtype=np.float32)
        return super().predict(X)
