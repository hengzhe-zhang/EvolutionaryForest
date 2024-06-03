import copy
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import r2_score, balanced_accuracy_score

from evolutionary_forest.utility.metric.visualization import (
    average_loss,
    calculate_ambiguity,
)

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class MTLTestFunction:
    def __init__(self, x, y, regr: "EvolutionaryForestRegressor", number_of_tasks):
        self.x = x
        self.y = y
        self.regr = regr
        self.number_of_tasks = number_of_tasks

    def predict_loss(self):
        if len(self.x) > 0:
            y_p = self.regr.predict(self.x)
            r2_scores = []

            for i in range(self.number_of_tasks):
                y_task = self.y[:, i]
                y_p_task = y_p[:, i]
                r2_scores.append(r2_score(y_task, y_p_task))

            return np.mean(r2_scores)
        else:
            return 0

    def __deepcopy__(self, memodict={}):
        return copy.deepcopy(self)


class TestFunction:
    def __init__(self, x, y, regr: "EvolutionaryForestRegressor" = None):
        self.x = x
        self.y = y
        self.regr = regr

    def predict_loss(self):
        if len(self.x) > 0:
            y_p = self.regr.predict(self.x)
            if isinstance(self.regr, ClassifierMixin):
                return balanced_accuracy_score(self.y, y_p)
            else:
                return r2_score(self.y, y_p)
        else:
            return 0

    def __deepcopy__(self, memodict={}):
        return copy.deepcopy(self)


class TestDiversity:
    def __init__(self, test_function, regr: "EvolutionaryForestRegressor"):
        self.test_function: TestFunction = test_function
        self.regr = regr

    def calculate_diversity(self, population):
        predictions_all = []
        for ind in population:
            features = self.regr.feature_generation(self.test_function.x, ind)
            prediction = ind.pipe.predict(features)
            predictions_all.append(prediction)
        predictions_all = np.array(predictions_all)
        return average_loss(predictions_all, self.test_function.y), calculate_ambiguity(
            predictions_all, self.test_function.y
        )
