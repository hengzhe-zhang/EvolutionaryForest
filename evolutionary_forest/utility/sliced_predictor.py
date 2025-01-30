from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class SlicedPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, step_size=50000, verbose=False):
        """
        A wrapper for regressors to slice data during prediction.

        Parameters:
        - regressor: The regressor to wrap.
        - step_size: The number of samples per slice. Default is 50000.
        - verbose: If True, print progress messages. Default is False.
        """
        self.regressor = regressor
        self.step_size = step_size
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        if self.verbose:
            print("Fitting the regressor...")
        self.regressor.fit(X, y, **fit_params)
        if self.verbose:
            print("Fitting completed.")
        return self  # Return the fitted instance

    def predict(self, X, **predict_params):
        if self.verbose:
            print(f"Predicting with {len(X)} samples...")

        # Check if data needs to be sliced
        if len(X) > self.step_size:
            slices = [
                X[i : i + self.step_size] for i in range(0, len(X), self.step_size)
            ]
            results = []

            for idx, slice_data in enumerate(slices):
                if self.verbose:
                    print(
                        f"Processing slice {idx + 1}/{len(slices)} with {len(slice_data)} samples..."
                    )
                result = self.regressor.predict(slice_data, **predict_params)
                results.append(result)

            combined_results = np.concatenate(results)
            if self.verbose:
                print("Prediction completed.")
            return combined_results
        else:
            if self.verbose:
                print("Processing without slicing...")
            return self.regressor.predict(X, **predict_params)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    # Create sample data
    X, y = make_regression(n_samples=100000, n_features=10, noise=0.1)

    # Wrap a LinearRegression model
    lr = LinearRegression()
    # Fit the model
    lr.fit(X[:100], y[:100])

    wrapped_regressor = SlicedPredictor(lr, step_size=10000)

    # Make predictions, automatically slicing large data
    predictions = wrapped_regressor.predict(X)
    print(predictions.shape)
