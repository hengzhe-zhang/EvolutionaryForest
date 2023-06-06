from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from deap.tools import selNSGA2, sortNondominated
from numpy.linalg import norm

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor

import numpy as np


def distance_to_hyperplane(points, X):
    points = np.array(points)
    X = np.array(X)
    A = points[0]
    vectors = points[1:] - A

    # Compute the SVD
    U, S, V = np.linalg.svd(vectors)

    # Obtain the normal vector from the last column of V
    N = V[-1]

    # Calculate the projection of X onto the hyperplane
    t = np.dot(A - X, N) / np.dot(N, N)
    P = X + t * N

    # Compute the distance from X to the projection point P
    distance = np.linalg.norm(X - P)
    return distance


def knee_point_detection(front):
    # For maximization problem
    front = np.array(front)
    p_array = []
    for dim in range(front.shape[1]):
        id = np.argmax(front[:, dim])
        p_array.append(front[id])
    p_array = np.array(p_array)
    ans = np.argmax([distance_to_hyperplane(p_array, x) for x in front])
    # p1 = np.array([max(front[:, 0]), min(front[:, 1])])
    # p2 = np.array([min(front[:, 0]), max(front[:, 1])])
    # # 自动选择拐点
    # ans = max([i for i in range(len(front))],
    #           key=lambda i: norm(np.cross(p2 - p1, p1 - front[i])) / norm(p2 - p1))
    return ans


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
                 **kwargs):
        self.algorithm = algorithm
        self.objective_function = objective_function
        self.normalization = normalization
        self.knee_point = knee_point

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

        population[:] = selNSGA2(individuals, len(population))

        if self.knee_point:
            first_pareto_front = sortNondominated(population, len(population))[0]
            knee = knee_point_detection([p.fitness.wvalues for p in first_pareto_front])
            # Select the knee point as the final model
            self.algorithm.hof = [first_pareto_front[knee]]

        if self.objective_function != None:
            self.objective_function.restore(individuals)
        return population


if __name__ == "__main__":
    points = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]

    X = [1, 1, 1, 1, 1]

    distance = distance_to_hyperplane(points, X)
    print("The distance between point X and the hyperplane is:", distance)
