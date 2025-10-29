import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from evolutionary_forest.model.knn.FaissKNNRegressor import FaissKNNRegressor


class AutoKNNRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible KNN regressor that automatically selects
    the best k using cross-validation on the full training data.
    """

    def __init__(self, k_list=(3, 5, 10, 20, 30), cv=5, scoring=None, n_jobs=1):
        self.k_list = k_list
        self.cv = cv
        self.scoring = scoring or make_scorer(r2_score)
        self.n_jobs = n_jobs
        self.best_k_ = None
        self.best_val_score_ = None
        self.model_ = None

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        best_score, best_k = -np.inf, None

        for k in self.k_list:
            model = FaissKNNRegressor(n_neighbors=k)
            scores = cross_val_score(
                model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
            )
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score, best_k = mean_score, k

        self.best_k_ = best_k
        self.best_val_score_ = best_score
        print("Best k:", self.best_k_, "with CV score:", self.best_val_score_)

        # If model worse than baseline, use zero-prediction
        if self.best_val_score_ is None or self.best_val_score_ <= 0:
            self.model_ = DummyRegressor(strategy="constant", constant=0.0)
            self.best_k_ = None  # not applicable
        else:
            self.model_ = FaissKNNRegressor(n_neighbors=self.best_k_)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = AutoKNNRegressor(k_list=[3, 5, 10, 20, 30])
    model.fit(X, y)
    print(model.best_k_, model.best_val_score_)
