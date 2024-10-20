import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


class RFRoutingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_estimators, n_estimators_rf=10):
        self.base_estimators_ = base_estimators
        self.n_estimators_rf = n_estimators_rf

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        predictions = np.array([est.predict(X) for est in self.base_estimators_]).T
        errors = np.square(y[:, np.newaxis] - predictions)
        best_args = errors.argmin(axis=1)
        self.candidates_ = np.unique(best_args)
        self.router_ = RandomForestClassifier(n_estimators=self.n_estimators_rf)
        self.router_.fit(X, best_args)
        self.base_predictions_ = predictions
        return self

    def predict(self, X):
        check_is_fitted(
            self, ["base_estimators_", "router_", "candidates_", "base_predictions_"]
        )
        X = np.array(X)
        predictions = np.array([est.predict(X) for est in self.base_estimators_])
        proba = self.router_.predict_proba(X)
        relevant_proba = proba[:, self.candidates_]
        selected_predictions = predictions[self.candidates_, :]
        final_prediction = np.sum(relevant_proba * selected_predictions.T, axis=1)
        return final_prediction


if __name__ == "__main__":
    # Generate some sample data
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = (
        X[:, 0] * 5 + X[:, 1] * 3 + np.random.randn(100) * 0.5
    )  # linear combination with noise

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pre-train the base estimators
    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    gb = GradientBoostingRegressor(n_estimators=50, random_state=0)

    rf.fit(X_train, y_train)  # Fit RandomForest on training data
    gb.fit(X_train, y_train)  # Fit GradientBoosting on training data

    # Define pre-trained base estimators for the Dynamic Selection Ensemble
    base_estimators = [rf, gb]

    # Initialize RFRoutingEnsemble
    ensemble = RFRoutingEnsemble(base_estimators=base_estimators, n_estimators_rf=10)

    # Fit the ensemble model
    ensemble.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble.predict(X_test)

    # Calculate performance
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
