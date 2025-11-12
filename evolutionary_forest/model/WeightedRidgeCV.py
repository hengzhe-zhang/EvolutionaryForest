from sklearn.linear_model import RidgeCV


class WeightedRidgeCV(RidgeCV):
    def set_instance_weights(self, weights):
        self.instance_weights = weights

    def fit(self, X, y, sample_weight=None, **params):
        if sample_weight is None and self.instance_weights is not None:
            sample_weight = self.instance_weights
        return super().fit(X, y, sample_weight=sample_weight)
