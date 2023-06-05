from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from deap.tools import selNSGA2
from numpy.linalg import norm

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def knee_point_detection(front):
    # For maximization problem
    p1 = np.max(front, axis=0)
    p2 = np.max(front, axis=0)
    # 自动选择拐点
    ans = max([i for i in range(len(front))],
              key=lambda i: norm(np.cross(p2 - p1, p1 - front[i])) / norm(p2 - p1))
    return ans


class EnvironmentalSelection():
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
                 objective: Objective = None,
                 normalization=False,
                 knee_point=False):
        self.algorithm = algorithm
        self.objective = objective
        self.normalization = normalization
        self.knee_point = knee_point

    def select(self, population, offspring):
        individuals = population + offspring
        if self.objective != None:
            self.objective.set(individuals)

        if self.normalization:
            dims = len(individuals[0].values)
            for d in range(dims):
                values = [ind.values[d] for ind in individuals]
                min_val = min(values)
                max_val = max(values)
                for ind in individuals:
                    ind.values[d] = (ind.values[d] - min_val) / (max_val - min_val)

        population[:] = selNSGA2(individuals, len(population))

        if self.knee_point:
            knee = knee_point_detection([p.fitness.wvalues for p in population])
            # Select the knee point as the final model
            self.hof = [population[knee]]

        if self.objective != None:
            self.objective.restore(individuals)
        return population
