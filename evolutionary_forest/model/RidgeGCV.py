import numpy as np
from sklearn.linear_model._ridge import RidgeCV
from sklearn.base import _fit_context
from sklearn.utils.validation import check_is_fitted


class RidgeGCV(RidgeCV):
    """RidgeCV with LOOCV predictions at selected alpha."""

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None, **params):
        if self.cv is not None:
            raise ValueError("RidgeGCV only supports cv=None (LOOCV / GCV mode).")
        orig_store, orig_scoring = self.store_cv_results, self.scoring
        try:
            self.store_cv_results = True
            if self.scoring is None:
                # Force `_RidgeGCV` to compute/store per-sample LOO predictions.
                def _neg_mse(_est, y_pred, y_true, **_):
                    return -np.mean((y_true - y_pred) ** 2)

                self.scoring = _neg_mse

            super().fit(X, y, sample_weight=sample_weight, **params)

            alphas = np.asarray(self.alphas, dtype=float).ravel()

            def _alpha_idx(a):
                a = float(a)
                m = np.flatnonzero(alphas == a)
                return int(m[0]) if m.size else int(np.argmin(np.abs(alphas - a)))

            cvp = self.cv_results_
            alpha_ = self.alpha_
            # Convert to array to check shape
            alpha_arr = np.asarray(alpha_)
            if alpha_arr.ndim == 0:
                # Scalar case
                self.loocv_predictions_ = cvp[..., _alpha_idx(float(alpha_arr))]
            else:
                # Multi-output case
                idxs = [_alpha_idx(a) for a in alpha_arr.ravel()]
                self.loocv_predictions_ = np.column_stack(
                    [cvp[:, j, idxs[j]] for j in range(len(idxs))]
                )

            if not orig_store:
                del self.cv_results_
        finally:
            self.store_cv_results, self.scoring = orig_store, orig_scoring
        return self

    def predict_loocv(self):
        check_is_fitted(self, attributes=["loocv_predictions_"])
        return self.loocv_predictions_
