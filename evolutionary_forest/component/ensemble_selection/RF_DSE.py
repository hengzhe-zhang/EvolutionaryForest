import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.multiclass import OneVsRestClassifier


class RFRoutingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(
        self, base_estimators, n_estimators_rf=10, random_state=None, **rf_kwargs
    ):
        # Note: trailing underscore is usually for fitted attrs, but keeping your naming.
        self.base_estimators_ = base_estimators
        self.n_estimators_rf = n_estimators_rf
        self.random_state = random_state
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # preds: (n_samples, n_estimators)
        preds = np.column_stack([est.predict(X) for est in self.base_estimators_])

        # For each sample, which base estimator had the lowest squared error?
        best_labels = np.argmin(
            (y[:, None] - preds) ** 2, axis=1
        )  # labels are estimator indices

        # Train a classifier to route to the best estimator index
        self.router_ = RandomForestClassifier(
            n_estimators=self.n_estimators_rf,
            random_state=self.random_state,
            **self.rf_kwargs,
        )
        self.router_.fit(X, best_labels)

        # Store the classes (subset of estimator indices) and keep their order
        self.classes_ = self.router_.classes_  # e.g., array([3, 7, 34, ...])
        return self

    def predict(self, X):
        check_is_fitted(self, ["router_", "classes_"])
        X = np.asarray(X)

        # base_preds: (n_estimators, n_samples)
        base_preds = np.vstack([est.predict(X) for est in self.base_estimators_])

        # Select only the estimators the router knows about, in the SAME order as classes_
        # selected: (k, n_samples), where k = len(self.classes_)
        selected = base_preds[self.classes_, :]

        # proba: (n_samples, k) with columns aligned to router_.classes_
        proba = self.router_.predict_proba(X)

        # Weighted average over candidates
        # (n_samples, k) * (n_samples, k) -> sum over k -> (n_samples,)
        final_prediction = np.sum(proba * selected.T, axis=1)
        return final_prediction


class TopKRoutingEnsemble(RFRoutingEnsemble):
    """
    Inherits from RFRoutingEnsemble and routes using Top-K multi-label classification:
      - For each sample, mark the K lowest-error base estimators as positives.
      - Train OvR(RandomForestClassifier) on this indicator.
      - Blend base predictions with the OvR probabilities (optionally normalized).
    """

    def __init__(
        self,
        base_estimators,
        n_estimators_rf=10,
        random_state=None,
        *,
        k=10,
        normalize="softmax",
        min_pos_per_class=1,
        **rf_kwargs,
    ):
        """
        normalize: 'softmax' | 'sum' | None
        min_pos_per_class: drop estimators that never appear in anyone's top-K
        """
        super().__init__(base_estimators, n_estimators_rf, random_state, **rf_kwargs)
        self.k = k
        self.normalize = normalize
        self.min_pos_per_class = min_pos_per_class

    @staticmethod
    def _softmax(P):
        Z = P - P.max(axis=1, keepdims=True)
        np.exp(Z, out=Z)
        Z_sum = Z.sum(axis=1, keepdims=True) + 1e-12
        return Z / Z_sum

    def _normalize_weights(self, P):
        if self.normalize == "softmax":
            return self._softmax(P)
        elif self.normalize == "sum":
            return P / (P.sum(axis=1, keepdims=True) + 1e-12)
        return P

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Reuse base estimators from parent
        base_preds = np.column_stack(
            [est.predict(X) for est in self.base_estimators_]
        )  # (n, M)
        errors = (y[:, None] - base_preds) ** 2

        M = base_preds.shape[1]
        K = min(self.k, M)

        # Build Top-K indicator
        topk_idx = np.argpartition(errors, K - 1, axis=1)[:, :K]  # (n, K)
        Y_bin = np.zeros((X.shape[0], M), dtype=int)
        Y_bin[np.arange(X.shape[0])[:, None], topk_idx] = 1

        # Keep only estimators that have at least min_pos_per_class positives
        pos_counts = Y_bin.sum(axis=0)
        keep_mask = pos_counts >= self.min_pos_per_class
        kept = np.flatnonzero(keep_mask)

        # Fallback: ensure at least one kept class
        if kept.size == 0:
            best1 = np.argmin(errors, axis=1)
            Y_bin = np.zeros((X.shape[0], M), dtype=int)
            Y_bin[np.arange(X.shape[0]), best1] = 1
            kept = np.unique(best1)
            keep_mask = np.zeros(M, dtype=bool)
            keep_mask[kept] = True

        Y_bin_kept = Y_bin[:, keep_mask]

        # Train OvR router (reusing RF config from parent)
        base_router = RandomForestClassifier(
            n_estimators=self.n_estimators_rf,
            random_state=self.random_state,
            **self.rf_kwargs,
        )
        self.router_ = OneVsRestClassifier(base_router)
        self.router_.fit(X, Y_bin_kept)

        # Store metadata to align probabilities back to estimator indices
        self.kept_classes_ = (
            kept  # estimator indices kept (order = columns of Y_bin_kept)
        )
        self.n_estimators_total_ = M
        return self

    def predict(self, X):
        check_is_fitted(self, ["router_", "kept_classes_", "n_estimators_total_"])
        X = np.asarray(X)

        # Base predictions for all estimators (n, M)
        base_preds = np.column_stack([est.predict(X) for est in self.base_estimators_])

        # OvR probabilities for kept classes only (n, K_kept); column order matches Y_bin_kept
        P_kept = self.router_.predict_proba(X)

        # Map back to full M with zeros for dropped classes
        n = X.shape[0]
        M = self.n_estimators_total_
        P_full = np.zeros((n, M), dtype=float)
        P_full[:, self.kept_classes_] = P_kept

        # Normalize to mixture weights and blend
        W = self._normalize_weights(P_full)  # (n, M)
        y_hat = np.sum(W * base_preds, axis=1)
        return y_hat


if __name__ == "__main__":
    # Generate some sample data
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = (
        X[:, 0] * 5 + X[:, 1] * 3 + np.random.randn(100) * 0.5
    )  # linear combination with noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pre-train the base estimators
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    gb = GradientBoostingRegressor(n_estimators=50, random_state=0)

    rf.fit(X_train, y_train)  # Fit RandomForest on training data
    gb.fit(X_train, y_train)  # Fit GradientBoosting on training data

    # Define pre-trained base estimators for the Dynamic Selection Ensemble
    base_estimators = [rf, gb]

    # Initialize RFRoutingEnsemble
    ensemble = RFRoutingEnsemble(base_estimators=base_estimators, n_estimators_rf=10)

    # Fit the ensemble model
    ensemble.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble.predict(X_test)

    # Calculate performance
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
