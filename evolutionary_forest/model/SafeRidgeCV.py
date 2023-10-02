import numpy as np
from sklearn.linear_model import RidgeCV


class BoundedRidgeCV(RidgeCV):
    def fit(self, X, y, sample_weight=None):
        self.min = min(y)
        self.max = max(y)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        return np.clip(super().predict(X), self.min, self.max)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)
