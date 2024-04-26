import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer

from evolutionary_forest.component.stgp.smooth_scaler import (
    NearestValueTransformer2D,
)


class SmoothRidgeCV(RidgeCV):
    def fit(self, X, y, sample_weight=None):
        self.transformer = NearestValueTransformer2D()
        X = self.transformer.fit_transform(X)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        X = self.transformer.transform(X)
        return super().predict(X)


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


class SplineRidgeCV(RidgeCV):
    def fit(self, X, y, sample_weight=None):
        self.spline = SplineTransformer()
        X = self.spline.fit_transform(X)
        return super().fit(X, y, sample_weight)

    def predict(self, X):
        X = self.spline.transform(X)
        return super().predict(X)
