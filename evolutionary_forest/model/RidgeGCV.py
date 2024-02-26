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
