import numpy as np
from sklearn.linear_model import RidgeCV


class BoundedRidgeCV(RidgeCV):
    def fit(self, X, y, sample_weight=None):
        self.min = min(y)
        self.max = max(y)
        result = super().fit(X, y, sample_weight)
        self.post_fix(y)
        return result

    def post_fix(self, y):
        """
        Try to clip cross-validation values
        """
        # change back
        self.cv_values_ += np.mean(y)
        self.cv_values_ = np.clip(self.cv_values_, self.min, self.max)
        self.cv_values_ -= np.mean(y)

    def predict(self, X):
        return np.clip(super().predict(X), self.min, self.max)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
