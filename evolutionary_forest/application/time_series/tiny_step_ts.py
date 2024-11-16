import math
import time

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression


def create_lagged_features(y, lag=1):
    X, y_lag = [], []
    for i in range(lag, len(y)):
        X.append(y[i - lag : i])  # Use the previous 'lag' values
        y_lag.append(y[i])  # The target is the next value
    return np.array(X), np.array(y_lag)


# Function to create lagged features for autoregressive forecasting
def compute_increments(y):
    return np.diff(y)  # Returns an array of increments


# Function to compute the difference (increment) between consecutive values
class AugmentedBaselineIncrementalForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, lag=1, model=None, verbose=False, augment_interval=0):
        """
        Initializes the BaselineIncrementalForecaster.

        Parameters:
        - lag (int): Number of lagged original steps to use as features.
        - model: A scikit-learn compatible regressor. If None, LinearRegression is used.
        - verbose (bool): If True, prints training and prediction info.
        - augment_interval (int): Number of interpolated points to insert between each original data point.
        """
        self.lag_original = lag
        self.model = model if model is not None else LinearRegression()
        self.verbose = verbose
        self.augment_interval = augment_interval
        self.k = augment_interval  # For clarity
        self.fitted_ = False  # Indicator if the model is fitted

        # Calculate the augmented lag
        if self.augment_interval > 0:
            self.lag_augmented = self.lag_original * (self.augment_interval + 1)
        else:
            self.lag_augmented = self.lag_original

    def fit(self, y, fh=None):
        """
        Fits the forecaster to the training data.

        Parameters:
        - y (array-like): The time series data to fit.
        - fh (array-like, optional): Forecast horizon. Not used in fit.

        Returns:
        - self
        """
        if self.verbose:
            print("[INFO] Training BaselineIncrementalForecaster...")
        start_time = time.time()

        # Augment the data if augment_interval > 0
        y_augmented = (
            self.augment_data(y, self.augment_interval)
            if self.augment_interval > 0
            else np.array(y)
        )
        if self.verbose:
            print(f"[DEBUG] y_augmented: {y_augmented}")

        # Compute increments on augmented data
        increments_augmented = self.compute_increments(y_augmented)
        if self.verbose:
            print(f"[DEBUG] increments_augmented: {increments_augmented}")

        # Verification Step 1: Reconstruct y_augmented from increments_augmented
        y_reconstructed_augmented = self.reconstruct_from_increments(
            y_augmented[0], increments_augmented
        )
        if self.verbose:
            print(f"[DEBUG] y_reconstructed_augmented: {y_reconstructed_augmented}")

        # Check if reconstruction of y_augmented is accurate
        if not np.allclose(y_reconstructed_augmented, y_augmented):
            raise ValueError(
                "Reconstruction from increments_augmented does not match y_augmented."
            )
        else:
            if self.verbose:
                print("[INFO] Increments successfully reconstruct y_augmented.")

        # Verification Step 2: Reconstruct y_train from increments_augmented
        if self.k > 0:
            # Calculate the number of original increments
            original_increment_count = math.ceil(
                len(increments_augmented) / (self.k + 1)
            )
            y_reconstructed = [y_augmented[0]]
            for i in range(original_increment_count):
                start_idx = i * (self.k + 1)
                end_idx = start_idx + (self.k + 1)
                group = increments_augmented[start_idx:end_idx]
                sum_increments = np.sum(group)
                y_reconstructed.append(y_reconstructed[-1] + sum_increments)
            y_reconstructed = np.array(y_reconstructed)
            if self.verbose:
                print(f"[DEBUG] y_reconstructed from y_train: {y_reconstructed}")

            # Compare to original y
            y_input = np.array(y)
            if not np.allclose(y_reconstructed, y_input):
                raise ValueError(
                    "Reconstruction from increments_augmented does not match the original y_train."
                )
            else:
                if self.verbose:
                    print(
                        "[INFO] Increments successfully reconstruct the original y_train."
                    )
        else:
            # If augment_interval=0, y_augmented == y_train, so no need to reconstruct
            y_reconstructed = y_reconstructed_augmented
            if self.verbose:
                print("[INFO] augment_interval=0; y_augmented equals y_train.")

        # Create lagged features using the augmented lag
        X_train_aug, y_train_aug = self.create_lagged_features(
            increments_augmented, self.lag_augmented
        )
        if self.verbose:
            print(f"[DEBUG] X_train_aug shape: {X_train_aug.shape}")
            print(f"[DEBUG] y_train_aug shape: {y_train_aug.shape}")

        # Fit the model
        self.model.fit(X_train_aug, y_train_aug)
        self.fitted_ = True

        if self.verbose:
            print(
                f"[INFO] BaselineIncrementalForecaster trained in {time.time() - start_time:.2f} seconds."
            )
            print(
                f"[INFO] Training data size: {len(y_augmented)} points with {len(increments_augmented)} increments."
            )

        return self

    def predict(self, y_train, fh):
        """
        Predicts future values based on the trained model.

        Parameters:
        - y_train (array-like): The historical time series data.
        - fh (int or array-like): Forecast horizon (number of original steps to predict).

        Returns:
        - y_pred (np.array): Predicted future values.
        """
        if not self.fitted_:
            raise NotFittedError(
                "This BaselineIncrementalForecaster instance is not fitted yet. Call 'fit' first."
            )

        if self.verbose:
            print("[INFO] Predicting with BaselineIncrementalForecaster...")
        start_time = time.time()

        # Ensure fh is an integer representing the number of original steps to predict
        if isinstance(fh, (list, np.ndarray)):
            fh_steps = len(fh)
        elif isinstance(fh, int):
            fh_steps = fh
        else:
            raise ValueError("fh must be an integer or a list/array-like of integers.")

        # Total augmented steps to predict
        total_augmented_steps = fh_steps * (self.k + 1)

        # Augment the data if augment_interval > 0
        y_train = (
            self.augment_data(y_train, self.augment_interval)
            if self.augment_interval > 0
            else np.array(y_train)
        )

        # Compute increments from y_train
        increments_train = self.compute_increments(y_train)

        # Initialize last_window with the last 'lag_augmented' increments
        if len(increments_train) >= self.lag_augmented:
            last_window = list(increments_train[-self.lag_augmented :])
        else:
            # If not enough increments, pad with zeros
            padding = [0] * (self.lag_augmented - len(increments_train))
            last_window = padding + list(increments_train)

        if self.verbose:
            print(f"[DEBUG] Initial last_window: {last_window}")

        # List to store all predicted augmented increments
        y_pred_increments_augmented = []

        # Predict total_augmented_steps increments
        for step in range(total_augmented_steps):
            # Predict next increment using the current window
            next_increment = self.model.predict(np.array([last_window]))[0]

            # Handle `inf` or `nan` predictions by replacing them with the last valid increment
            if np.isnan(next_increment) or np.isinf(next_increment):
                next_increment = last_window[
                    -1
                ]  # Use the last valid increment as a fallback
                if self.verbose:
                    print(
                        "[WARNING] Detected inf/nan prediction; replaced with last valid increment."
                    )

            y_pred_increments_augmented.append(next_increment)

            if self.verbose:
                print(
                    f"[DEBUG] Step {step + 1}: Predicted increment = {next_increment}"
                )

            # Update the window with the new predicted increment
            last_window = last_window[1:] + [next_increment]

        # Group the predicted increments into fh_steps groups of (k + 1)
        y_pred_increments_original = [
            sum(y_pred_increments_augmented[i * (self.k + 1) : (i + 1) * (self.k + 1)])
            for i in range(fh_steps)
        ]

        if self.verbose:
            print(f"[DEBUG] y_pred_increments_original: {y_pred_increments_original}")

        # Reconstruct the actual predictions by accumulating the summed increments
        last_actual = y_train[-1]
        y_pred = [
            last_actual + np.sum(y_pred_increments_original[: i + 1])
            for i in range(len(y_pred_increments_original))
        ]

        # Ensure no `nan` or `inf` values remain in the final output
        y_pred = np.nan_to_num(
            y_pred, nan=last_actual, posinf=last_actual, neginf=last_actual
        )

        if self.verbose:
            print(
                f"[INFO] Prediction completed in {time.time() - start_time:.2f} seconds."
            )
            print(f"[INFO] Forecasted {fh_steps} original steps.")

        return np.array(y_pred)

    def augment_data(self, y, augment_interval):
        """
        Performs linear interpolation to augment the training data.

        Parameters:
        - y (array-like): Original time series data.
        - augment_interval (int): Number of interpolated points to insert between each original point.

        Returns:
        - y_augmented (np.array): Augmented time series data.
        """
        y = np.array(y)
        if len(y) < 2:
            raise ValueError(
                "Time series must have at least two points to perform interpolation."
            )

        if augment_interval <= 0:
            return y

        # Number of new points between each original point
        k = augment_interval
        y_augmented = []

        for i in range(len(y) - 1):
            y_augmented.append(y[i])
            # Linear interpolation
            for j in range(1, k + 1):
                interp_value = y[i] + (y[i + 1] - y[i]) * j / (k + 1)
                y_augmented.append(interp_value)

        y_augmented.append(y[-1])  # Append the last original point

        if self.verbose:
            print(
                f"[INFO] Data augmented with interval {augment_interval}. Original size: {len(y)}, Augmented size: {len(y_augmented)}"
            )

        return np.array(y_augmented)

    def compute_increments(self, y):
        """
        Computes the increments (differences) of the time series.

        Parameters:
        - y (array-like): Time series data.

        Returns:
        - increments (np.array): Differences between consecutive data points.
        """
        y = np.array(y)
        increments = np.diff(y)
        return increments

    def reconstruct_from_increments(self, first_value, increments):
        """
        Reconstructs the time series from increments.

        Parameters:
        - first_value (float): The starting value of the time series.
        - increments (array-like): The increments to reconstruct the series.

        Returns:
        - y_reconstructed (np.array): The reconstructed time series.
        """
        y_reconstructed = [first_value]
        for inc in increments:
            y_reconstructed.append(y_reconstructed[-1] + inc)
        return np.array(y_reconstructed)

    def create_lagged_features(self, increments, lag):
        """
        Creates lagged features from the increments.

        Parameters:
        - increments (array-like): Increments of the time series.
        - lag (int): Number of lagged increments to use as features.

        Returns:
        - X (np.array): Feature matrix.
        - y (np.array): Target vector.
        """
        increments = np.array(increments)
        if len(increments) < lag + 1:
            raise ValueError(
                f"Not enough data points to create lagged features with lag={lag}."
            )

        X = []
        y = []
        for i in range(lag, len(increments)):
            X.append(increments[i - lag : i])
            y.append(increments[i])
        return np.array(X), np.array(y)


class BaselineIncrementalForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, lag=1, model=None, verbose=False):
        self.lag = lag
        self.model = model
        self.verbose = verbose

    def fit(self, y, fh=None):
        if self.verbose:
            print("[INFO] Training BaselineIncrementalForecaster...")
        start_time = time.time()

        # Calculate the increments (differences)
        increments = compute_increments(y)

        # Create lagged features using increments
        X_train, y_train = create_lagged_features(increments, self.lag)

        # Fit the model
        self.model.fit(X_train, y_train)

        if self.verbose:
            print(
                f"[INFO] BaselineIncrementalForecaster trained in {time.time() - start_time:.2f} seconds."
            )
        return self

    def predict(self, y_train, fh):
        if self.verbose:
            print("[INFO] Predicting with BaselineIncrementalForecaster...")
        start_time = time.time()

        # Calculate the increments (differences)
        increments = compute_increments(y_train)

        # Recursive forecasting to predict increments
        y_pred_increments = []
        last_window = list(
            increments[-self.lag :]
        )  # Start with the last 'lag' increments

        for _ in fh:
            # Predict next increment
            next_pred = self.model.predict(np.array([last_window]))[0]

            # Handle `inf` or `nan` predictions by replacing them with the last valid increment
            if np.isnan(next_pred) or np.isinf(next_pred):
                next_pred = last_window[
                    -1
                ]  # Use the last valid increment as a fallback
                if self.verbose:
                    print(
                        "[WARNING] Detected inf/nan prediction; replaced with last valid increment."
                    )

            y_pred_increments.append(next_pred)

            # Update the window with the new predicted increment
            last_window = last_window[1:] + [next_pred]

        # Reconstruct the actual predictions by accumulating the increments
        last_actual = y_train[-1]
        y_pred = [
            last_actual + np.sum(y_pred_increments[: i + 1])
            for i in range(len(y_pred_increments))
        ]

        # Ensure no `nan` or `inf` values remain in the final output
        y_pred = np.nan_to_num(
            y_pred, nan=last_actual, posinf=last_actual, neginf=last_actual
        )

        if self.verbose:
            print(
                f"[INFO] Prediction completed in {time.time() - start_time:.2f} seconds."
            )
        return np.array(y_pred)


# Scikit-Learn compatible Decomposed Incremental Forecaster
class DecomposedIncrementalForecaster(BaseEstimator, RegressorMixin):
    def __init__(self, lag=1, scale_factor=0.1, model=None, verbose=False):
        self.lag = lag
        self.scale_factor = scale_factor
        self.model = model
        self.verbose = verbose

    def fit(self, y, fh=None):
        if self.verbose:
            print("[INFO] Training DecomposedIncrementalForecaster...")
        start_time = time.time()

        # Calculate the increments (differences)
        increments = compute_increments(y)

        # Create lagged features using increments
        X_train, y_train = create_lagged_features(increments, self.lag)

        # Fit the model
        self.model.fit(X_train, y_train)

        if self.verbose:
            print(
                f"[INFO] DecomposedIncrementalForecaster trained in {time.time() - start_time:.2f} seconds."
            )
        return self

    def predict(self, y_train, fh):
        if self.verbose:
            print("[INFO] Predicting with DecomposedIncrementalForecaster...")
        start_time = time.time()

        # Calculate the increments (differences)
        increments = compute_increments(y_train)

        # Decomposed prediction logic to predict increments
        y_pred_increments = []
        last_window_large = list(increments[-self.lag :])  # Initialize large window

        # Determine the number of steps based on scale_factor
        if self.scale_factor <= 0 or self.scale_factor > 1:
            raise ValueError(
                "scale_factor must be between 0 (exclusive) and 1 (inclusive)."
            )
        steps = max(int(1 / self.scale_factor), 1)

        for _ in fh:
            # Initialize the current increment based on the last increment in the large window
            current_increment = last_window_large[-1]

            # Initialize the small window with the current large window
            last_window_small = last_window_large.copy()

            # Decompose the prediction into smaller steps
            for _ in range(steps):
                # Predict the next increment based on the small window
                next_pred = self.model.predict([last_window_small])[0]

                # Apply scaled prediction for the increment
                current_increment += (next_pred - current_increment) * self.scale_factor

                # Update the small window with the new current increment
                last_window_small = last_window_small[1:] + [current_increment]

            # Store the final predicted increment after all small steps
            y_pred_increments.append(current_increment)

            # Update the large window with the new predicted increment
            last_window_large = last_window_large[1:] + [current_increment]

        # Reconstruct the actual predictions by accumulating the increments
        last_actual = y_train[-1]
        y_pred = [
            last_actual + np.sum(y_pred_increments[: i + 1])
            for i in range(len(y_pred_increments))
        ]

        if self.verbose:
            print(
                f"[INFO] Prediction completed in {time.time() - start_time:.2f} seconds."
            )
        return np.array(y_pred)
