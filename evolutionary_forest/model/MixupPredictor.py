import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from evolutionary_forest.model.RidgeGCV import RidgeGCV


class MixupRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn wrapper to implement mixup prediction for any regressor.

    Parameters:
    regressor: scikit-learn regressor object
        The regressor to wrap.
    sigma: float, default=0.1
        The noise level used for generating perturbations.
    q: int, default=10
        The number of perturbed pairs to generate for each test point.
    random_state: int or RandomState, optional
        The generator used to initialize the noise.
    """

    def __init__(self, regressor, sigma=0.01, q=10, random_state=None):
        self.regressor = regressor
        self.sigma = sigma
        self.q = q
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Fit the regressor
        self.regressor_ = self.regressor.fit(X, y)
        if isinstance(self.regressor_, (RidgeCV, RidgeGCV)):
            self.alphas = self.regressor_.alphas
            self.alpha_ = self.regressor_.alpha_
            self.coef_ = self.regressor_.coef_
            self.cv_results_ = self.regressor_.cv_results_
        return self

    def predict(self, X: np.ndarray):
        # Check is fit had been called
        check_is_fitted(self, "regressor_")
        # Input validation
        X: np.ndarray = check_array(X)

        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        predictions = np.zeros((self.q, n_samples))

        for i in range(self.q):
            # Generate perturbed samples
            noise = random_state.normal(0, self.sigma, size=X.shape)
            X_plus = X + noise
            X_minus = X - noise

            # Predict on perturbed samples
            predictions[i] = (
                self.regressor_.predict(X_plus) + self.regressor_.predict(X_minus)
            ) / 2

        # Average over all perturbed predictions
        return np.mean(predictions, axis=0)


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # Create some sample data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize the mixup regressor with a linear regression model
    mixup_regressor = MixupRegressor(
        regressor=KNeighborsRegressor(n_neighbors=3), sigma=0.1, q=10, random_state=0
    )
