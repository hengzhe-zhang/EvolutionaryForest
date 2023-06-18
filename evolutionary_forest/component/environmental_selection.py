from abc import abstractmethod
from typing import TYPE_CHECKING

from deap.tools import selNSGA2, sortNondominated, selSPEA2
from numpy.linalg import norm
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from evolutionary_forest.multigene_gp import multiple_gene_compile, result_calculation

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor

import numpy as np


def knee_point_detection(front):
    ht = HighTradeoffPoints()
    # convert to minimization
    ans = ht.do(-1 * np.array(front))
    return max(ans, key=lambda x: front[x][0])


class EnvironmentalSelection():
    @abstractmethod
    def select(self, population, offspring):
        pass


class Objective():
    @abstractmethod
    def set(self, individuals):
        pass

    def restore(self, individuals):
        for ind in individuals:
            ind.fitness.weights = (-1,)
            ind.fitness.values = getattr(ind, 'original_fitness')


class TreeSizeObjective(Objective):
    def __init__(self):
        pass

    def set(self, individuals):
        for ind in individuals:
            setattr(ind, 'original_fitness', ind.fitness.values)
            ind.fitness.weights = (-1, -1)
            ind.fitness.values = (ind.fitness.values[0], np.sum([len(y) for y in ind.gene]))


class NSGA2(EnvironmentalSelection):

    def __init__(self,
                 algorithm: "EvolutionaryForestRegressor",
                 objective_function: Objective = None,
                 normalization=False,
                 knee_point=False,
                 bootstrapping_selection=False,
                 **kwargs):
        self.bootstrapping_selection = bootstrapping_selection
        self.algorithm = algorithm
        self.objective_function = objective_function
        self.normalization = normalization
        self.knee_point = knee_point
        self.selection_operator = selNSGA2

    def select(self, population, offspring):
        individuals = population + offspring
        if self.objective_function != None:
            self.objective_function.set(individuals)

        if self.normalization:
            dims = len(individuals[0].fitness.values)
            min_max = []
            for d in range(dims):
                values = [ind.fitness.values[d] for ind in individuals]
                min_val = min(values)
                max_val = max(values)
                min_max.append((min_val, max_val))
            for ind in individuals:
                values = []
                for d in range(dims):
                    min_val, max_val = min_max[d]
                    values.append((ind.fitness.values[d] - min_val) / (max_val - min_val))
                ind.fitness.values = values

        population[:] = self.selection_operator(individuals, len(population))

        if self.knee_point:
            first_pareto_front = sortNondominated(population, len(population))[0]
            knee = knee_point_detection([p.fitness.wvalues for p in first_pareto_front])
            # Select the knee point as the final model
            self.algorithm.hof = [first_pareto_front[knee]]

        if self.bootstrapping_selection:
            first_pareto_front: list = sortNondominated(population, len(population))[0]

            def quick_evaluation(ind):
                r2_scores = []
                func = multiple_gene_compile(ind, self.algorithm.pset)
                for train_index, test_index in KFold(shuffle=True, random_state=0).split(self.algorithm.X,
                                                                                         self.algorithm.y):
                    Yp = result_calculation(func, self.algorithm.X, False)
                    ind.pipe.fit(Yp[train_index], self.algorithm.y[train_index])
                    r2_scores.append(r2_score(self.algorithm.y[test_index], ind.pipe.predict(Yp[test_index])))
                return np.mean(r2_scores)

            # Select the minimal cross-validation error as the final model
            self.algorithm.hof = [max(first_pareto_front, key=quick_evaluation)]

        if self.objective_function != None:
            self.objective_function.restore(individuals)
        return population


class SPEA2(NSGA2):

    def __init__(self, algorithm: "EvolutionaryForestRegressor", objective_function: Objective = None,
                 normalization=False, knee_point=False, bootstrapping_selection=False, **kwargs):
        super().__init__(algorithm, objective_function, normalization, knee_point, bootstrapping_selection, **kwargs)
        self.selection_operator = selSPEA2


if __name__ == "__main__":
    pass
