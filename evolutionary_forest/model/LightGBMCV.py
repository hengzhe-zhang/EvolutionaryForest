import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


class LightGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_estimators=int(1e10),
        learning_rate=0.1,
        early_stopping_rounds=10,
        test_size=0.2,
        random_state=None,
        **kwargs,
    ):
        """
        Custom sklearn-style LightGBM regressor.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.random_state = random_state
        self.kwargs = kwargs
        self.model_ = None

    def fit(self, X, y):
        """
        Fit the LightGBM model.
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Initialize the LightGBM regressor
        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            **self.kwargs,
        )

        # Define the early stopping callback
        early_stopping_callback = lgb.early_stopping(
            stopping_rounds=self.early_stopping_rounds, verbose=False
        )

        # Fit the model with early stopping
        self.model_.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="l2",  # L2 loss (mean squared error)
            callbacks=[early_stopping_callback],
        )

        return self

    def predict(self, X):
        """
        Predict regression values for samples in X.
        """
        if self.model_ is None:
            raise ValueError(
                "The model is not fitted yet. Call 'fit' before 'predict'."
            )
        return self.model_.predict(X)

    def score(self, X, y):
        """
        Return the R^2 score on the given data.
        """
        if self.model_ is None:
            raise ValueError("The model is not fitted yet. Call 'fit' before 'score'.")
        return self.model_.score(X, y)


if __name__ == "__main__":
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=0)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Custom LightGBM with early stopping
    regressor_early_stop = LightGBMRegressor(verbose=-1)
    regressor_early_stop.fit(X_train, y_train)
    y_pred_early_stop = regressor_early_stop.predict(X_test)
    r2_score_early_stop = r2_score(y_test, y_pred_early_stop)

    # LightGBM without early stopping
    model_no_early_stop = lgb.LGBMRegressor(n_estimators=1000, verbose=-1)
    model_no_early_stop.fit(X_train, y_train)
    y_pred_no_early_stop = model_no_early_stop.predict(X_test)
    r2_score_no_early_stop = r2_score(y_test, y_pred_no_early_stop)
    print(f"R^2 score with early stopping: {r2_score_early_stop}")
    print(f"R^2 score without early stopping: {r2_score_no_early_stop}")
