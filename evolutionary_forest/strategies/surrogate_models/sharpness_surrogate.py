from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class GaussianProcessSharpness:
    def __init__(self):
        self.sharpness_prediction_model = Pipeline(
            [
                ("Scaler", StandardScaler()),
                ("GP", GaussianProcessRegressor()),
            ]
        )

    def fit(self, parent: List):
        x = []
        y = []
        for p in parent:
            semantics = p.case_values[:5]
            x.append(semantics)
            sharpness_value = p.fitness_list[1][0]
            y.append(sharpness_value)
        x = np.array(x)
        y = np.array(y)
        self.sharpness_prediction_model.fit(x, y)

    def predict(self, X):
        return self.sharpness_prediction_model.predict(X)

    def check_feasibility(self, individual, historical_best_score):
        sharpness_prediction_model = self.sharpness_prediction_model
        one_sample = individual.case_values[:5].reshape(1, -1)
        y_pred, sigma = sharpness_prediction_model.predict(one_sample, return_std=True)
        confidence_multiplier = 1.96
        # the lower bound of sharpness
        lower_bound = y_pred - confidence_multiplier * sigma
        base_score = np.mean(individual.case_values)
        if historical_best_score < base_score + lower_bound[0]:
            return False
        return True
