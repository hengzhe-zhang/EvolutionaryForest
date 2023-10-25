import numpy as np
from sklearn.linear_model import LogisticRegression


class SafetyLogisticRegression(LogisticRegression):
    def predict_proba(self, X):
        return np.nan_to_num(super().predict_proba(X))
