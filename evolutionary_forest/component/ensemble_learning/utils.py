from scipy import stats
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn2pmml.ensemble import GBDTLRClassifier


class GBDTLRClassifierX(GBDTLRClassifier):
    def fit(self, X, y, **fit_params):
        super().fit(X, y, **fit_params)
        self.classes_ = self.gbdt_.classes_
        return self


class EnsembleRegressor(RegressorMixin, BaseEstimator):
    """
    Combining several models generated by 5-fold CV to form a final ensemble model
    """

    def __init__(self, trees):
        self.trees = trees

    def fit(self, X, y):
        pass

    def predict(self, X):
        predictions = []
        t: DecisionTreeRegressor
        for t in self.trees:
            predictions.append(t.predict(X))
        return np.mean(predictions, axis=0)


class EnsembleClassifier(ClassifierMixin, BaseEstimator):
    """
    Combining several models generated by 5-fold CV to form a final ensemble model
    """

    def __init__(self, trees):
        self.trees = trees

    def fit(self, X, y):
        pass

    def predict(self, X):
        predictions = []
        t: DecisionTreeClassifier
        for t in self.trees:
            predictions.append(t.predict(X))
        return stats.mode(predictions, axis=0)[0].flatten()
