from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import r2_score

from evolutionary_forest.component.pac_bayesian import assign_rank
from evolutionary_forest.component.rademacher_complexity import generate_rademacher_vector, \
    rademacher_complexity_estimation
from evolutionary_forest.component.vc_dimension import vc_dimension_estimation
from evolutionary_forest.multigene_gp import MultipleGeneGP

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class Fitness():
    def fitness_value(self, individual, estimators, Y, y_pred):
        # very simple fitness evaluation
        return -1 * r2_score(Y, y_pred),

    @abstractmethod
    def post_processing(self, parent, population, hall_of_fame, elite_archive):
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
    def __init__(self, algorithm: "EvolutionaryForestRegressor", rademacher_mode='Analytical', **params):
        self.algorithm = algorithm
        self.size_objective = False
        self.historical_best_bounded_complexity = None
        self.historical_best_bounded_complexity_list = None
        self.rademacher_mode = rademacher_mode

    def fitness_value(self, individual, estimators, Y, y_pred):
        # very simple fitness evaluation
        return -1 * r2_score(Y, y_pred),

    def assign_complexity(self, individual, estimator):
        algorithm = self.algorithm
        # Extract features from input data based on individual
        X_features = algorithm.feature_generation(algorithm.X, individual)
        y = algorithm.y
        # Calculate R2 score, Rademacher complexity and Rademacher complexity list
        estimation, bounded_rademacher, bounded_rademacher_list = \
            rademacher_complexity_estimation(X_features, y, estimator,
                                             generate_rademacher_vector(algorithm.X),
                                             self.historical_best_bounded_complexity_list,
                                             algorithm.pac_bayesian,
                                             self.rademacher_mode)
        # Store results in individual's fitness list
        weighted_rademacher = estimation[1]
        individual.fitness_list = self.get_fitness_list(individual, weighted_rademacher[0])
        # Normalize mean squared error
        normalize_factor = np.mean((np.mean(y) - y) ** 2)
        if algorithm.pac_bayesian.bound_reduction:
            bounded_mse = np.mean(np.clip(individual.case_values / normalize_factor, 0, 1))
        else:
            bounded_mse = np.mean(individual.case_values / normalize_factor)
        if algorithm.pac_bayesian.bound_reduction or algorithm.pac_bayesian.direct_reduction:
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

    def assign_complexity_pop(self, pop):
        algorithm = self.algorithm
        y = algorithm.y
        normalize_factor = np.mean((np.mean(y) - y) ** 2)
        if algorithm.pac_bayesian.bound_reduction:
            reduced_evaluation = 0
            for p in pop:
                # Calculate bounded MSE
                bounded_mse = np.mean(np.clip(p.case_values / normalize_factor, 0, 1))
                # Update best bounded complexity if needed
                if self.historical_best_bounded_complexity is None or bounded_mse < self.historical_best_bounded_complexity:
                    self.assign_complexity(p, p.pipe)
                else:
                    p.fitness_list = self.get_fitness_list(p, np.inf)
                    reduced_evaluation += 1
            if algorithm.verbose:
                print('reduced_evaluation: ', reduced_evaluation)
        # If a complexity estimation ratio < 1 is used
        elif algorithm.pac_bayesian.complexity_estimation_ratio < 1:
            self.skip_based_on_threshold(algorithm, pop)
        else:
            for p in pop:
                self.assign_complexity(p, p.pipe)

    def get_fitness_list(self, individual, rademacher_complexity):
        algorithm = self.algorithm
        if self.size_objective:
            # Calculate tree size
            tree_size = sum([len(tree) for tree in individual.gene])
            return [(individual.fitness.wvalues[0], 1), (rademacher_complexity, algorithm.pac_bayesian.objective),
                    (tree_size, algorithm.pac_bayesian.objective)]
        else:
            return [(individual.fitness.wvalues[0], 1), (rademacher_complexity, algorithm.pac_bayesian.objective)]

    def skip_based_on_threshold(self, algorithm, pop):
        # Get R2 score threshold
        q = np.quantile([p.fitness.wvalues[0] for p in pop],
                        q=1 - algorithm.pac_bayesian.complexity_estimation_ratio)
        reduced_evaluation = 0
        for p in pop:
            # If fitness is better than q
            if p.fitness.wvalues[0] > q:
                self.assign_complexity(p, p.pipe)
            else:
                p.fitness_list = self.get_fitness_list(p, np.inf)
                reduced_evaluation += 1

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        reassign_objective_values(parent, population)


def reassign_objective_values(parent, pop):
    if parent != None:
        pop = parent + pop
    valid_components = [p.fitness_list[1][0] for p in pop
                        if p.fitness_list[1][0] < np.inf and not np.isnan(p.fitness_list[1][0])]
    if len(valid_components) == 0:
        max_rademacher = np.inf
    else:
        max_rademacher = max(valid_components)
    for individual in pop:
        individual.fitness.weights = tuple(-1 for _ in range(len(individual.fitness_list)))
        # R2 should be maximized, other should be minimized
        r2 = [-1 * individual.fitness_list[0][0]]
        rademacher = list(map(lambda x: x[0], individual.fitness_list[1:]))
        if rademacher[0] >= np.inf or np.isnan(rademacher[0]):
            # do not use inf because this will make NSGA-2 problematic
            rademacher[0] = max_rademacher + 1
        individual.fitness.values = tuple(r2 + rademacher)


class LocalRademacherComplexityR2(RademacherComplexityR2):

    def get_fitness_list(self, individual, rademacher_complexity):
        algorithm = self.algorithm
        weights = individual.pipe['Ridge'].coef_
        return [
            (individual.fitness.wvalues[0], 1),
            (rademacher_complexity, algorithm.pac_bayesian.objective),
            (np.linalg.norm(weights, ord=2), algorithm.pac_bayesian.objective),
        ]


class RademacherComplexitySizeR2(RademacherComplexityR2):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        super().__init__(algorithm, **params)
        self.size_objective = True


class RademacherComplexityR2Scaler(RademacherComplexityR2):
    """
    Aggregating fitness values into a scalar value
    """

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        assign_rank(population, hall_of_fame, elite_archive)


class LocalRademacherComplexityR2Scaler(LocalRademacherComplexityR2):
    """
    Aggregating fitness values into a scalar value
    """

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        self.assign_complexity_pop(population)
        assign_rank(population, hall_of_fame, elite_archive)


class TikhonovR2(Fitness):

    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        coef_norm = np.linalg.norm(y_pred) ** 2
        return (-1 * score, coef_norm)


class VCDimensionR2(Fitness):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        self.algorithm = algorithm
        self.size_objective = False

    def assign_complexity(self, individual, estimator):
        # reducing the time of estimating VC-Dimension
        algorithm = self.algorithm
        X_features = algorithm.feature_generation(algorithm.X, individual)
        feature_generator = partial(algorithm.feature_generation, individual=individual)
        y = algorithm.y

        # VCD estimation
        gene_length = sum([len(g) for g in individual.gene])
        estimation = vc_dimension_estimation(X_features, y, estimator,
                                             input_dimension=algorithm.X.shape[1],
                                             estimated_vcd=gene_length,
                                             feature_generator=feature_generator,
                                             optimal_design=algorithm.pac_bayesian.optimal_design)
        individual.fitness_list = estimation
        return -1 * individual.fitness_list[0][0],

    def assign_complexity_pop(self, pop):
        # get minimum r2
        ratio = self.algorithm.pac_bayesian.complexity_estimation_ratio
        q = np.quantile([p.fitness.wvalues[0] for p in pop],
                        q=1 - ratio)

        reduced_evaluation = 0
        for p in pop:
            if p.fitness.wvalues[0] > q:
                # better fitness value
                self.assign_complexity(p, p.pipe)
            else:
                p.fitness_list = [(p.fitness.wvalues[0], 1), (np.inf, -1)]
                reduced_evaluation += 1


class R2Size(Fitness):
    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        tree_size = sum([len(tree) for tree in individual.gene])
        return (-1 * score, tree_size)


class R2FeatureCount(Fitness):
    def fitness_value(self, individual: MultipleGeneGP, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        return (-1 * score, individual.gene_num)


class R2SizeScaler(Fitness):
    def __init__(self, algorithm: "EvolutionaryForestRegressor", **params):
        self.algorithm = algorithm

    def fitness_value(self, individual, estimators, Y, y_pred):
        score = r2_score(Y, y_pred)
        tree_size = sum([len(tree) for tree in individual.gene])
        individual.fitness_list = ((score, 1), (tree_size, -self.algorithm.pac_bayesian.objective))
        return -1 * score,

    def post_processing(self, parent, population, hall_of_fame, elite_archive):
        assign_rank(population, hall_of_fame, elite_archive)
