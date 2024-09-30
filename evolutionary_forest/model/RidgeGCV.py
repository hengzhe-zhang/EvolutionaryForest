import numpy as np
from sklearn.linear_model._ridge import _RidgeGCV


class RidgeGCV(_RidgeGCV):
    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        cast_to_ndarray=True,
        **check_params
    ):
        check_params["dtype"] = [np.float32, np.float64]
        return super()._validate_data(
            X, y, reset, validate_separately, cast_to_ndarray, **check_params
        )

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        if self.store_cv_values:
            self.cv_values_ = np.nan_to_num(
                self.cv_values_, nan=0.0, posinf=0.0, neginf=0.0
            )
        return self

    def predict(self, X):
        prediction = super().predict(X)
        prediction=np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return prediction

