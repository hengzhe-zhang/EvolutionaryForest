import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, Lasso
from sklearn.multioutput import MultiOutputRegressor

from evolutionary_forest.model.RidgeGCV import RidgeGCV


class MTLRidgeCV(RidgeCV):
    def __init__(self):
        super().__init__()
        self.mtl_ridge = MultiOutputRegressor(RidgeGCV(store_cv_results=True))
        self.coef_ = None

    def fit(self, X, y=None):
        if len(X) != len(y):
            y = np.reshape(y, (len(X), -1))
        self.mtl_ridge.fit(X, y)
        self.coef_ = np.mean([e.coef_ for e in self.mtl_ridge.estimators_], axis=0)
        self.cv_results_ = np.concatenate(
            [e.cv_results_ for e in self.mtl_ridge.estimators_], axis=0
        )
        return self

    def predict(self, X, y=None):
        return self.mtl_ridge.predict(X)

    def cv_prediction(self, y):
        tasks = len(self.mtl_ridge.estimators_)
        predictions = []
        for y_true, model in zip(y.reshape((-1, tasks)).T, self.mtl_ridge.estimators_):
            # RidgeGCV stores predictions directly in cv_predictions_
            if hasattr(model, "cv_predictions_"):
                real_p = model.cv_predictions_
            else:
                # Fallback: extract from cv_results_ for best alpha
                best_alpha_idx = tuple(model.alphas).index(model.alpha_)
                real_p = model.cv_results_[:, best_alpha_idx]
            predictions.append(real_p)
        return np.array(predictions).T


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
