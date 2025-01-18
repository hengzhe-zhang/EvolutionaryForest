from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from evolutionary_forest.component.ensemble_learning.DREP import DREPEnsemble
from evolutionary_forest.multigene_gp import result_calculation

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class StackingStrategy:
    def __init__(self, algorithm: "EvolutionaryForestRegressor") -> None:
        super().__init__()
        self.algorithm = algorithm

    def stacking_layer_generation(self, X, y):
        algorithm = self.algorithm
        # Check if a second layer is specified
        if algorithm.second_layer == "None" or algorithm.second_layer == None:
            return

        # Collect predictions from base models
        y_data = algorithm.y
        predictions = []
        for individual in algorithm.hof:
            predictions.append(individual.predicted_values)
        predictions = np.array(predictions)

        if algorithm.second_layer == "DREP":
            ensemble = DREPEnsemble(y_data, predictions, algorithm)
            ensemble.generate_ensemble_weights()
            algorithm.tree_weight = ensemble.get_ensemble_weights()
        elif algorithm.second_layer == "Ridge":
            # Ridge regression for generating ensemble weights
            algorithm.ridge = Ridge(alpha=1e-3, normalize=True, fit_intercept=False)
            algorithm.ridge.fit(predictions.T, y_data)
            x = algorithm.ridge.coef_
            x[x < 0] = 0
            x[x > 0] = 1
            x /= np.sum(x)
            x = x.flatten()
            algorithm.tree_weight = x
        elif algorithm.second_layer == "Ridge-Prediction":
            predictions = algorithm.individual_prediction(X)
            # fitting predicted values
            algorithm.ridge = RidgeCV(fit_intercept=False)
            algorithm.ridge.fit(predictions.T, y_data)
            algorithm.tree_weight = algorithm.ridge.coef_.flatten()
        elif algorithm.second_layer == "TreeBaseline":
            base_line_score = np.mean(cross_val_score(DecisionTreeRegressor(), X, y))
            score = np.array(
                list(map(lambda x: x.fitness.wvalues[0], algorithm.hof.items))
            )
            x = np.zeros_like(score)
            x[score > base_line_score] = 1
            if np.sum(x) == 0:
                x[0] = 1
            x /= np.sum(x)
            x = x.flatten()
            algorithm.tree_weight = x
        elif algorithm.second_layer == "DiversityPrune":
            sample_len = 500
            predictions = []
            for individual in algorithm.hof:
                func = algorithm.toolbox.compile(individual)
                x = np.random.randn(sample_len, X.shape[1])
                Yp = result_calculation(func, x, algorithm.original_features)
                predicted = individual.pipe.predict(Yp)
                predictions.append(predicted)
            predictions = np.array(predictions)

            # forward selection
            current_prediction = np.zeros(sample_len)
            remain_ind = set([i for i in range(len(algorithm.hof))])

            # select first regressor
            min_index = 0
            remain_ind.remove(min_index)

            ensemble_list = np.zeros(len(algorithm.hof))
            ensemble_list[min_index] = 1
            while True:
                div_list = []
                for i in remain_ind:
                    diversity = np.mean(((current_prediction - predictions[i]) ** 2))
                    div_list.append((diversity, i))
                div_list = list(sorted(div_list, key=lambda x: -x[0]))
                index = div_list[0][1]
                ensemble_size = np.sum(ensemble_list)
                trial_prediction = (
                    ensemble_size / (ensemble_size + 1) * current_prediction
                    + 1 / (ensemble_size + 1) * predictions[index]
                )
                if np.mean(((current_prediction - trial_prediction) ** 2)) < 0.05:
                    break
                current_prediction = trial_prediction
                ensemble_list[index] = 1
                remain_ind.remove(index)
            ensemble_list /= np.sum(ensemble_list)
            algorithm.tree_weight = ensemble_list
        elif algorithm.second_layer == "CAWPE":
            pop = algorithm.hof
            weight = np.ones(len(pop))
            for i, ind in enumerate(pop):
                fitness = ind.fitness.wvalues[0]
                if isinstance(algorithm, ClassifierMixin):
                    # using MSE as the default fitness criterion
                    weight[i] = (1 / fitness) ** 4
                else:
                    # using R^2 as the default fitness criterion
                    if fitness > 0:
                        weight[i] = (fitness) ** 4
                    else:
                        weight[i] = 0
            algorithm.tree_weight = weight / np.sum(weight)
