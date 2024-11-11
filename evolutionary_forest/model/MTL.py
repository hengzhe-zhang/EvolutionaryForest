import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, Lasso
from sklearn.metrics import r2_score, make_scorer
from sklearn.multioutput import MultiOutputRegressor

from evolutionary_forest.utils import cv_prediction_from_ridge


class MTLRidgeCV(RidgeCV):
    def __init__(self):
        super().__init__()
        self.mtl_ridge = MultiOutputRegressor(
            RidgeCV(store_cv_results=True, scoring=make_scorer(r2_score))
        )
        self.coef_ = None

    def fit(self, X, y=None):
        if len(X) != len(y):
            y = np.reshape(y, (len(X), -1))
        self.mtl_ridge.fit(X, y)
        self.coef_ = np.mean([e.coef_ for e in self.mtl_ridge.estimators_], axis=0)
        self.cv_values_ = np.concatenate(
            [e.cv_values_ for e in self.mtl_ridge.estimators_], axis=0
        )
        return self

    def predict(self, X, y=None):
        return self.mtl_ridge.predict(X)

    def cv_prediction(self, y):
        tasks = len(self.mtl_ridge.estimators_)
        predictions = []
        for y_true, model in zip(y.reshape((-1, tasks)).T, self.mtl_ridge.estimators_):
            real_p = cv_prediction_from_ridge(y_true, model)
            predictions.append(real_p)
        return np.concatenate(np.array(predictions).T, axis=0)


class MTLLassoCV(LassoCV):
    def __init__(self):
        super().__init__()
        self.mtl_lasso = MultiOutputRegressor(Lasso())
        self.coef_ = None

    def fit(self, X, y=None):
        if len(X) != len(y):
            y = np.reshape(y, (len(X), -1))
        self.mtl_lasso.fit(X, y)
        self.coef_ = np.mean([e.coef_ for e in self.mtl_lasso.estimators_], axis=0)
        return self

    def predict(self, X, y=None):
        return self.mtl_lasso.predict(X)
