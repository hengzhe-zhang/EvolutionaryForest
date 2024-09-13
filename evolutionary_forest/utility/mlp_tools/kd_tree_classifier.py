import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KDTree


class SemanticKDTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.kd_tree = None
        self.kd_tree_targets = None

    def fit(self, train_data):
        kd_tree_data = []
        kd_tree_targets = []

        for tree, semantics in train_data:
            kd_tree_data.append(semantics)
            kd_tree_targets.append(tree)
            kd_tree_data.append(-semantics)
            kd_tree_targets.append(tree)

        self.kd_tree = KDTree(np.array(kd_tree_data))
        self.kd_tree_targets = kd_tree_targets

        return self

    def predict(self, test_data):
        if self.kd_tree is None:
            raise RuntimeError("You must fit the model before predicting.")

        distances, indices = self.kd_tree.query(test_data, k=1)

        predictions = [self.kd_tree_targets[idx[0]] for idx in indices]

        return predictions
