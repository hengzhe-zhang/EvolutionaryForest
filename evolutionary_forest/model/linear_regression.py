import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV


class BoundedRidgeRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.ridge = RidgeCV()
        self.ridge.fit(X, y)
        self.y_train_ = y
        return self

    def predict(self, X):
        y_pred = self.ridge.predict(X)
        y_pred_bounded = np.clip(y_pred, np.min(self.y_train_), np.max(self.y_train_))
        return y_pred_bounded
