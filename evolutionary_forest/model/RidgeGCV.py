import numpy as np
from sklearn.linear_model._ridge import _RidgeGCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
        if self.store_cv_results:
            self.cv_values_ = np.nan_to_num(
                self.cv_values_, nan=0.0, posinf=0.0, neginf=0.0
            )
        return self

    def predict(self, X):
        prediction = super().predict(X)
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return prediction


def compare_ridge_gcv_with_ridgecv():
    # Create a random regression problem
    np.random.seed(0)
    X = np.random.randn(100, 10)
    y = X.dot(np.random.randn(10)) + np.random.randn(100) * 0.5

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define alphas (regularization strengths) to test
    alphas = np.logspace(-6, 6, 13)

    # Use the custom RidgeGCV class
    ridge_gcv = RidgeGCV(alphas=alphas, store_cv_results=True)
    ridge_gcv.fit(X_train, y_train)
    y_pred_custom = ridge_gcv.predict(X_test)

    # Use the standard RidgeCV class from sklearn
    ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
    ridge_cv.fit(X_train, y_train)
    y_pred_sklearn = ridge_cv.predict(X_test)

    # Calculate mean squared error for both
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)

    # Output comparison results
    print("Mean Squared Error (Custom RidgeGCV):", mse_custom)
    print("Mean Squared Error (sklearn RidgeCV):", mse_sklearn)

    # Check consistency between the two implementations
    assert np.allclose(
        y_pred_custom, y_pred_sklearn
    ), "Predictions differ between custom RidgeGCV and sklearn RidgeCV."
    assert np.allclose(
        mse_custom, mse_sklearn
    ), "MSE differs between custom RidgeGCV and sklearn RidgeCV."
    print("Results are consistent between custom RidgeGCV and sklearn RidgeCV.")


if __name__ == "__main__":
    # Run the comparison
    compare_ridge_gcv_with_ridgecv()
