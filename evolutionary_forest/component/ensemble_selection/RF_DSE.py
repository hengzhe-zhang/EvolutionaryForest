import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


class RFRoutingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(
        self, base_estimators, n_estimators_rf=10, random_state=None, **rf_kwargs
    ):
        # Note: trailing underscore is usually for fitted attrs, but keeping your naming.
        self.base_estimators_ = base_estimators
        self.n_estimators_rf = n_estimators_rf
        self.random_state = random_state
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # preds: (n_samples, n_estimators)
        preds = np.column_stack([est.predict(X) for est in self.base_estimators_])

        # For each sample, which base estimator had the lowest squared error?
        best_labels = np.argmin(
            (y[:, None] - preds) ** 2, axis=1
        )  # labels are estimator indices

        # Train a classifier to route to the best estimator index
        self.router_ = RandomForestClassifier(
            n_estimators=self.n_estimators_rf,
            random_state=self.random_state,
            **self.rf_kwargs,
        )
        self.router_.fit(X, best_labels)

        # Store the classes (subset of estimator indices) and keep their order
        self.classes_ = self.router_.classes_  # e.g., array([3, 7, 34, ...])
        return self

    def predict(self, X):
        check_is_fitted(self, ["router_", "classes_"])
        X = np.asarray(X)

        # base_preds: (n_estimators, n_samples)
        base_preds = np.vstack([est.predict(X) for est in self.base_estimators_])

        # Select only the estimators the router knows about, in the SAME order as classes_
        # selected: (k, n_samples), where k = len(self.classes_)
        selected = base_preds[self.classes_, :]

        # proba: (n_samples, k) with columns aligned to router_.classes_
        proba = self.router_.predict_proba(X)

        # Weighted average over candidates
        # (n_samples, k) * (n_samples, k) -> sum over k -> (n_samples,)
        final_prediction = np.sum(proba * selected.T, axis=1)
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
