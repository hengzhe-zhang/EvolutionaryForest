from .forest import EvolutionaryForestRegressor


class AdaptiveSelectionRegressor(EvolutionaryForestRegressor):
    def fit(self, X, y, test_X=None, categorical_features=None):
        if 0.8 * X.shape[0] < 50:
            self.select = "lexicase"
        return super().fit(X, y, test_X, categorical_features)
