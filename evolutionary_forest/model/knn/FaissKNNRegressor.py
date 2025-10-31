import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.neighbors import RadiusNeighborsRegressor
import faiss


class AdaptiveRNRegressor(RadiusNeighborsRegressor):
    def __init__(self, radius=1.0, **kwargs):
        super().__init__(radius=radius, **kwargs)
        self.baseline_radius = radius

    def fit(self, X, y):
        self.radius = self.baseline_radius * np.sqrt(np.mean(X.var(axis=0)))
        return super().fit(X, y)


class FaissKNNRegressor(BaseEstimator, RegressorMixin):
    """Minimal FAISS-based KNN Regressor."""

    def __init__(self, n_neighbors=5, metric="l2", n_threads=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_threads = n_threads

    def _build_index(self, X):
        d = X.shape[1]

        faiss.omp_set_num_threads(self.n_threads)

        if self.metric == "l2":
            index = faiss.IndexFlatL2(d)
        elif self.metric == "ip":
            index = faiss.IndexFlatIP(d)
        else:
            raise ValueError("metric must be 'l2' or 'ip'")
        index.add(X)
        return index

    def fit(self, X, y):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        self._X = X
        self._y = y
        self.index_ = self._build_index(X)
        return self

    def _neighbors(self, X, k=None, include_self=False):
        """Return neighbor indices for given k."""
        k = k or self.n_neighbors
        k_query = k + 1 if include_self else k
        _, indices = self.index_.search(X.astype(np.float32), k_query)
        if include_self:
            indices = indices[:, 1 : k + 1]
        return indices

    def predict(self, X):
        indices = self._neighbors(X)
        return np.mean(self._y[indices], axis=1)


class FaissKNNRankRegressor(FaissKNNRegressor):
    """FAISS-based KNN Regressor with weighted predictions purely based on rank."""

    def __init__(self, n_neighbors=5, metric="l2", n_threads=1):
        super().__init__(n_neighbors, metric, n_threads)

    def predict(self, X):
        indices = self._neighbors(X)

        # Create inverse rank-based weights: rank 1 gets the highest weight (1), rank 2 gets 1/2, etc.
        ranks = np.arange(
            1, self.n_neighbors + 1, dtype=np.float32
        )  # Rank 1, 2, ..., k
        weights = 1 / ranks  # Inverse of the rank, rank 1 gets the highest weight

        # Normalize the weights so that they sum to 1 for each query point
        weights /= np.sum(weights)  # Normalize across the ranks

        # Weighted prediction using the neighbors' target values
        weighted_predictions = np.sum(weights * self._y[indices], axis=1)
        return weighted_predictions


class FaissKNNLinearRankRegressor(FaissKNNRegressor):
    """FAISS-based KNN Regressor with weighted predictions linearly decreasing based on rank."""

    def __init__(self, n_neighbors=5, metric="l2", n_threads=1):
        super().__init__(n_neighbors, metric, n_threads)

    def predict(self, X):
        indices = self._neighbors(X)

        # Create linearly decreasing weights: rank 1 gets n_neighbors, rank 2 gets n_neighbors-1, ..., rank k gets 1
        ranks = np.arange(
            self.n_neighbors, 0, -1, dtype=np.float32
        )  # Rank n_neighbors, n_neighbors-1, ..., 1
        weights = (
            ranks  # The weight is the rank, where the first gets the highest weight
        )

        # Normalize the weights so that they sum to 1 for each query point
        weights /= np.sum(weights)  # Normalize across the ranks

        # Weighted prediction using the neighbors' target values
        weighted_predictions = np.sum(weights * self._y[indices], axis=1)
        return weighted_predictions


class FaissKNNInverseDistanceRegressor(FaissKNNRegressor):
    """FAISS-based KNN Regressor with weighted predictions inversely based on distance."""

    def __init__(self, n_neighbors=5, metric="l2", n_threads=1):
        super().__init__(n_neighbors, metric, n_threads)

    def predict(self, X):
        indices = self._neighbors(X)

        # Calculate the distances from the neighbors
        distances, _ = self.index_.search(X.astype(np.float32), self.n_neighbors)

        # Inverse of the distance as weight
        weights = 1 / (
            distances + 1e-6
        )  # Adding small epsilon to avoid division by zero

        # Normalize weights to ensure they sum to 1 for each query point
        weights /= np.sum(weights, axis=1, keepdims=True)

        # Weighted prediction using the neighbors' target values
        weighted_predictions = np.sum(weights * self._y[indices], axis=1)
        return weighted_predictions


class RobustFaissKNNRegressor(FaissKNNRegressor):
    """FAISS KNN Regressor with LOO-R² K-selection and constant fallback."""

    def __init__(
        self, n_neighbors=30, metric="l2", n_threads=1, verbose=False, min_neighbors=1
    ):
        super().__init__(n_neighbors, metric, n_threads)
        self.verbose = verbose
        self.min_neighbors = min_neighbors

    def fit(self, X, y):
        super().fit(X, y)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        Kmax = self.n_neighbors
        Kmin = self.min_neighbors

        # --- Use inherited helper ---
        indices = self._neighbors(X, k=Kmax, include_self=True)
        Y_nn = self._y[indices]  # (N, Kmax)

        # --- Vectorized cumulative LOO evaluation ---
        cumsum = np.cumsum(Y_nn, axis=1)
        Ks = np.arange(1, Kmax + 1, dtype=np.float32)
        mean_preds = cumsum / Ks

        residuals = y[:, None] - mean_preds
        mse = np.mean(residuals**2, axis=0)
        y_var = np.var(y)
        r2_per_k = 1 - mse / y_var

        # --- Pick best K within [Kmin, Kmax] ---
        valid_range = slice(Kmin - 1, Kmax)
        best_local = np.argmax(r2_per_k[valid_range])
        self.k_opt_ = int(Kmin + best_local)
        self.train_score_ = float(r2_per_k[self.k_opt_ - 1])

        if self.verbose:
            print(f"Optimal K={self.k_opt_} with LOO R²={self.train_score_:.4f}")

        # Fallback logic
        if self.train_score_ <= 0:
            self.constant_ = np.mean(y)
            self.use_constant_ = True
        else:
            self.use_constant_ = False

        return self

    def predict(self, X):
        if getattr(self, "use_constant_", False):
            return np.full((len(X),), self.constant_, dtype=np.float32)
        indices = self._neighbors(X, k=self.k_opt_)
        return np.mean(self._y[indices], axis=1)


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import r2_score
    from sklearn.neighbors import KNeighborsRegressor

    # Load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # ---- (1) Naive KNN ----
    t0 = time.time()
    knn_naive = KNeighborsRegressor(n_neighbors=5)
    knn_naive.fit(X_train, y_train)
    t1 = time.time()
    r2_naive = r2_score(y_test, knn_naive.predict(X_test))

    print(f"[Naive KNN] K=5 | R²(test)={r2_naive:.4f} | Fit time={t1 - t0:.3f}s")

    # ---- (2) GridSearchCV KNN ----
    t0 = time.time()
    param_grid = {"n_neighbors": range(1, 41)}
    knn_cv = GridSearchCV(
        KNeighborsRegressor(), param_grid, scoring="r2", cv=5, n_jobs=1
    )
    knn_cv.fit(X_train, y_train)
    t1 = time.time()
    r2_cv = r2_score(y_test, knn_cv.predict(X_test))
    best_k_cv = knn_cv.best_params_["n_neighbors"]

    print(
        f"[CV KNN] Best K={best_k_cv} | R²(test)={r2_cv:.4f} | Fit time={t1 - t0:.3f}s"
    )

    # ---- (3) RobustFaissKNNRegressor ----
    t0 = time.time()
    robust_knn = RobustFaissKNNRegressor(n_neighbors=30, verbose=False)
    robust_knn.fit(X_train, y_train)
    t1 = time.time()
    r2_robust = r2_score(y_test, robust_knn.predict(X_test))
    print(
        f"[LOO KNN] Best K={robust_knn.k_opt_} | R²(test)={r2_robust:.4f} | Fit time={t1 - t0:.3f}s"
    )
