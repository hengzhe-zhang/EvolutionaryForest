from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class SlicedPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, step_size=50000):
        """
        A wrapper for regressors to slice data during prediction.

        Parameters:
        - regressor: The regressor to wrap.
        - step_size: The number of samples per slice. Default is 50000.
        """
        self.regressor = regressor
        self.step_size = step_size

    def fit(self, X, y=None, **fit_params):
        # Fit the regressor to the data
        self.regressor.fit(X, y, **fit_params)
        return self  # Return the fitted instance

    def predict(self, X, **predict_params):
        # Check if data needs to be sliced
        if len(X) > self.step_size:
            # Split the data into slices
            slices = [
                X[i : i + self.step_size] for i in range(0, len(X), self.step_size)
            ]

            # Process each slice and collect results
            results = []
            for slice_data in slices:
                result = self.regressor.predict(slice_data, **predict_params)
                results.append(result)

            # Combine the results from all slices
            combined_results = np.concatenate(results)
            return combined_results
        else:
            # If data is within the step size, process it directly
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
