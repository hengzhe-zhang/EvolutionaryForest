from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict

import evolutionary_forest.forest as forest
from evolutionary_forest.component.fitness import Fitness
from evolutionary_forest.model.WKNN import GaussianKNNRegressor


class R2WKNN(Fitness):
    def __init__(self, algorithm: "forest.EvolutionaryForestRegressor", **params):
        super().__init__()
        self.algorithm = algorithm

    def fitness_value(self, individual, estimators, Y, y_pred):
        """
        GKNN for regularization
        """
        score = r2_score(Y, y_pred)
        predictions = cross_val_predict(
            GaussianKNNRegressor(k=15), self.algorithm.X, Y, cv=5
        )
        loss_on_original_learner = mean_squared_error(Y, y_pred)
        loss_on_gknn = mean_squared_error(Y, predictions)
        individual.sam_loss = loss_on_gknn + loss_on_original_learner
        return (-1 * score, loss_on_gknn)
