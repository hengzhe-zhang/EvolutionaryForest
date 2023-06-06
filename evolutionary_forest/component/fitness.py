from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from evolutionary_forest.component.rademacher_complexity import generate_rademacher_vector, \
    rademacher_complexity_estimation

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class Fitness():
    @abstractmethod
    def fitness_value(self, individual, estimators, Y, y_pred):
        pass


class MTLR2(Fitness):
    def __init__(self, number_of_tasks):
        self.number_of_tasks = number_of_tasks

    def fitness_value(self, individual, estimators, Y, y_pred):
        number_of_tasks = self.number_of_tasks
        Y_partitions = np.array_split(Y, number_of_tasks)
        y_pred_partitions = np.array_split(y_pred, number_of_tasks)

        r_squared_values = []

        for i in range(number_of_tasks):
            mean_y = np.mean(Y_partitions[i])
            tss = np.sum((Y_partitions[i] - mean_y) ** 2)
            rss = np.sum((Y_partitions[i] - y_pred_partitions[i]) ** 2)
            r_squared = 1 - (rss / tss)
            r_squared_values.append(r_squared)
        return -1 * np.mean(r_squared_values),


class RademacherComplexityR2(Fitness):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        self.algorithm = algorithm
        self.size_objective = False
        self.historical_best_bounded_complexity = None
        self.historical_best_bounded_complexity_list = None

    def fitness_value(self, individual, estimators, Y, y_pred):
        algorithm = self.algorithm
        # Extract features from input data based on individual
        X_features = algorithm.feature_generation(algorithm.X, individual)
        y = algorithm.y

        # Calculate R2 score, Rademacher complexity and Rademacher complexity list
        estimation, bounded_rademacher, bounded_rademacher_list = \
            rademacher_complexity_estimation(X_features, y, estimators[0],
                                             generate_rademacher_vector(algorithm.X),
                                             self.historical_best_bounded_complexity_list,
                                             algorithm.pac_bayesian.objective)

        # Store results in individual's fitness list
        if self.size_objective:
            # Also store individual size
            tree_size = sum([len(tree) for tree in individual.gene])
            individual.fitness_list = (estimation[0], estimation[1], (tree_size, algorithm.pac_bayesian.objective))
        else:
            individual.fitness_list = estimation

        # Normalize mean squared error
        normalize_factor = np.mean((np.mean(y) - y) ** 2)
        bounded_mse = np.mean(np.clip(individual.case_values / normalize_factor, 0, 1))

        if algorithm.pac_bayesian.bound_reduction:
            # Reduce training time based on the Rademacher bound
            current_bound = bounded_mse + 2 * bounded_rademacher
            current_bound_list = bounded_mse + 2 * np.array(bounded_rademacher_list)
            if self.historical_best_bounded_complexity is None:
                # Store historical best bound
                self.historical_best_bounded_complexity = current_bound
                self.historical_best_bounded_complexity_list = current_bound_list
            elif self.historical_best_bounded_complexity > current_bound:
                self.historical_best_bounded_complexity = current_bound
                self.historical_best_bounded_complexity_list = current_bound_list
        # assert abs(individual.fitness.wvalues[0] - individual.fitness_list[0][0]) <= 1e-6

        # R2 should be maximized, other should be minimized
        r2 = [-1 * individual.fitness_list[0][0]]
        rademacher = list(map(lambda x: x[0], individual.fitness_list[1:]))
        return tuple(r2 + rademacher)


class RademacherComplexitySizeR2(RademacherComplexityR2):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        super().__init__(algorithm)
        self.size_objective = True
