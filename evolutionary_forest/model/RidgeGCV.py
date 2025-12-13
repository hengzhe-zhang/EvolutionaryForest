import numpy as np
from sklearn.linear_model._ridge import _RidgeGCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y


class RidgeGCV(_RidgeGCV):
    """
    Efficient Ridge regression with Leave-One-Out Cross-Validation.

    This implementation stores actual LOO CV predictions in cv_results_
    (not MSE values). After fitting, cv_predictions_ contains the LOO CV
    predictions for the optimal alpha, which can be accessed directly.
    """

    def _validate_data(
        self,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        cast_to_ndarray=True,
        **check_params,
    ):
        check_params["dtype"] = [np.float32, np.float64]
        return super()._validate_data(
            X, y, reset, validate_separately, cast_to_ndarray, **check_params
        )

    def fit(self, X, y, sample_weight=None):
        # Store X and y for efficient LOO CV computation
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self._X_fit = X
        self._y_fit = y.copy()

        # Fit using parent class
        super().fit(X, y, sample_weight)

        # If store_cv_results, compute and store actual LOO predictions
        if self.store_cv_results:
            self._compute_loo_predictions()
            self.cv_results_ = np.nan_to_num(
                self.cv_results_, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Store best alpha's predictions for easy access
            self._store_best_cv_predictions()
        return self

    def _store_best_cv_predictions(self):
        """
        Store CV predictions for the best alpha in cv_predictions_ attribute.

        The predictions are stored automatically when store_cv_results=True,
        allowing direct access via the cv_predictions_ attribute.
        """
        if self.cv_results_ is None:
            return

        # Get optimal alpha index - use argmin to handle floating point precision
        alphas_array = np.asarray(self.alphas)
        best_alpha_idx = np.argmin(np.abs(alphas_array - self.alpha_))

        # Extract predictions for best alpha
        if self.cv_results_.ndim == 2:
            # Single output case: (n_samples, n_alphas)
            self.cv_predictions_ = self.cv_results_[:, best_alpha_idx]
        else:
            # Multi-output case: (n_samples, n_targets, n_alphas)
            self.cv_predictions_ = self.cv_results_[:, :, best_alpha_idx]

    def _compute_loo_predictions(self):
        """
        Efficiently compute Leave-One-Out CV predictions for all alphas.
        Uses the hat matrix formula: y_loo[i] = (y_pred[i] - y[i]*h[i,i]) / (1 - h[i,i])
        This is O(n*p^2) instead of O(n^2*p) for naive LOO CV.
        """
        X = self._X_fit
        y = self._y_fit

        # Center X and y if fit_intercept
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            # Handle both single and multi-output cases
            if y.ndim == 1:
                y_mean = y.mean()
            else:
                y_mean = y.mean(axis=0)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            # Initialize y_mean with correct shape for broadcasting
            if y.ndim == 1:
                y_mean = 0.0
            else:
                y_mean = np.zeros(y.shape[1])

        n_samples, n_features = X_centered.shape
        n_alphas = len(self.alphas)

        # Initialize cv_results_ to store predictions (not MSE)
        if y.ndim == 1:
            self.cv_results_ = np.zeros((n_samples, n_alphas))
        else:
            n_targets = y.shape[1]
            self.cv_results_ = np.zeros((n_samples, n_targets, n_alphas))

        # Compute LOO predictions for each alpha
        for alpha_idx, alpha in enumerate(self.alphas):
            # Tall case: use X^T X formulation
            XtX = X_centered.T @ X_centered
            reg_matrix = XtX + alpha * np.eye(n_features)
            try:
                inv_XtX = np.linalg.solve(reg_matrix, np.eye(n_features))
            except np.linalg.LinAlgError:
                inv_XtX = np.linalg.pinv(reg_matrix)

            # Hat matrix: H = X @ inv_XtX @ X^T
            # Diagonal: h[i,i] = X[i] @ inv_XtX @ X[i]^T
            hat_diag = np.sum((X_centered @ inv_XtX) * X_centered, axis=1)

            # Full model predictions
            coef = inv_XtX @ X_centered.T @ y_centered
            y_pred = X_centered @ coef

            # Compute LOO predictions using efficient formula
            # y_loo[i] = (y_pred[i] - y[i]*h[i,i]) / (1 - h[i,i])
            # Avoid division by zero when h[i,i] is close to 1
            hat_diag = np.clip(hat_diag, -np.inf, 1 - 1e-12)
            denominator = 1.0 - hat_diag

            if y.ndim == 1:
                y_pred_loo = (y_pred - y_centered * hat_diag) / denominator
                y_pred_loo = y_pred_loo + y_mean
                self.cv_results_[:, alpha_idx] = y_pred_loo
            else:
                hat_diag_expanded = hat_diag[:, np.newaxis]
                denominator_expanded = denominator[:, np.newaxis]
                y_pred_loo = (
                    y_pred - y_centered * hat_diag_expanded
                ) / denominator_expanded
                # Check shape compatibility for multi-output case
                y_mean_array = np.asarray(y_mean)
                if (
                    y_mean_array.ndim > 0
                    and y_mean_array.shape[0] != y_pred_loo.shape[1]
                ):
                    raise ValueError(
                        f"Shape mismatch for multi-output: y_mean has shape {y_mean_array.shape}, "
                        f"but y_pred_loo has shape {y_pred_loo.shape}. "
                        f"Expected y_mean.shape[0] == {y_pred_loo.shape[1]}"
                    )
                y_pred_loo = y_pred_loo + y_mean
                self.cv_results_[:, :, alpha_idx] = y_pred_loo

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
    assert np.allclose(y_pred_custom, y_pred_sklearn), (
        "Predictions differ between custom RidgeGCV and sklearn RidgeCV."
    )
    assert np.allclose(mse_custom, mse_sklearn), (
        "MSE differs between custom RidgeGCV and sklearn RidgeCV."
    )
    print("Results are consistent between custom RidgeGCV and sklearn RidgeCV.")


if __name__ == "__main__":
    # Run the comparison
    compare_ridge_gcv_with_ridgecv()
