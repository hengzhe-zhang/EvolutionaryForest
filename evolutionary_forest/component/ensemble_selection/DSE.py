import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted


class DynamicSelectionEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimators, n_neighbors=5):
        self.base_estimators = base_estimators
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        self.cached_predictions_ = []
        self.cached_errors_ = []

        for model in self.base_estimators:
            predictions = model.predict(self.X_train_)
            self.cached_predictions_.append(predictions)
            errors = np.square(self.y_train_ - predictions)
            self.cached_errors_.append(errors)
        self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh_.fit(self.X_train_)

        return self

    def _get_model_weights(self, indices):
        weights = []
        for err in self.cached_errors_:
            local_error = np.mean(err[indices])
            weight = 1 / (local_error + 1e-10)
            weights.append(weight)
        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights.sum()
        return weights

    def predict(self, X):
        check_is_fitted(self, ["X_train_", "y_train_", "cached_predictions_", "neigh_"])

        X = np.array(X)
        all_predictions = np.array([model.predict(X) for model in self.base_estimators])
        y_pred = np.zeros(X.shape[0])
        distances, indices = self.neigh_.kneighbors(
            X, n_neighbors=self.n_neighbors, return_distance=True
        )
        for i in range(X.shape[0]):
            weights = self._get_model_weights(indices[i])
            y_pred[i] = np.dot(weights, all_predictions[:, i])

        return y_pred


if __name__ == "__main__":
    # Load dataset
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Pre-train the base estimators
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    gb = GradientBoostingRegressor(n_estimators=50, random_state=0)

    rf.fit(X_train, y_train)  # Fit RandomForest on training data
    gb.fit(X_train, y_train)  # Fit GradientBoosting on training data

    # Define pre-trained base estimators for the Dynamic Selection Ensemble
    base_estimators = [rf, gb]

    # Create and train the Dynamic Selection Ensemble
    dynamic_ensemble = DynamicSelectionEnsemble(
        base_estimators=base_estimators, n_neighbors=5
    )
    dynamic_ensemble.fit(X_train, y_train)

    # Make predictions using Dynamic Selection Ensemble
    y_pred_des = dynamic_ensemble.predict(X_test)

    # Train a simple RandomForestRegressor for comparison
    rf_single = RandomForestRegressor(n_estimators=50, random_state=0)
    rf_single.fit(X_train, y_train)

    # Make predictions using RandomForest
    y_pred_rf = rf_single.predict(X_test)

    # Calculate RMSE for both models
    rmse_des = np.sqrt(mean_squared_error(y_test, y_pred_des))
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

    # Calculate R^2 score for both models
    r2_des = r2_score(y_test, y_pred_des)
    r2_rf = r2_score(y_test, y_pred_rf)

    # Output results
    print(f"RMSE (Dynamic Ensemble): {rmse_des:.4f}")
    print(f"RMSE (Random Forest): {rmse_rf:.4f}")
    print(f"R² (Dynamic Ensemble): {r2_des:.4f}")
    print(f"R² (Random Forest): {r2_rf:.4f}")
