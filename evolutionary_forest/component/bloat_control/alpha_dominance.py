import math
from typing import TYPE_CHECKING

import numpy as np
from deap.tools import sortNondominated, selNSGA2

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class AlphaDominance:
    def __init__(self, algorithm: "EvolutionaryForestRegressor"):
        self.historical_largest = 0
        self.historical_smallest = math.inf
        self.algorithm = algorithm

    def update_best(self, population):
        self.historical_smallest = min(
            self.historical_smallest, min([len(p) for p in population])
        )
        self.historical_largest = max(
            self.historical_largest, max([len(p) for p in population])
        )

    def selection(self, population, offspring, alpha):
        # Adjust fitness considering size
        self.set_fitness_with_size(population, offspring, alpha)

        # Apply NSGA-II selection
        first_pareto_front = sortNondominated(offspring + population, len(population))[
            0
        ]
        population[:] = selNSGA2(offspring + population, len(population))
        self.algorithm.hof = [max(population, key=lambda x: x.fitness.wvalues[0])]

        # Restore original fitness values
        self.restore_original_fitness(population)

        # Adjust alpha based on size
        theta = np.rad2deg(np.arctan(alpha))
        avg_size = np.mean([len(p) for p in first_pareto_front])
        alpha = self.adjust_alpha(theta, avg_size)
        return alpha

    def adjust_alpha(self, theta, avg_size):
        historical_largest = self.historical_largest
        historical_smallest = self.historical_smallest
        theta = theta + (
            historical_largest + historical_smallest - 2 * avg_size
        ) * self.algorithm.bloat_control["step_size"] / (
            historical_largest - historical_smallest
        )
        theta = np.clip(theta, 0, 90)
        return np.tan(np.deg2rad(theta))

    def restore_original_fitness(self, population):
        for ind in population:
            ind.fitness.weights = (-1,)
            ind.fitness.values = getattr(ind, "original_fitness")

    def set_fitness_with_size(self, population, offspring, alpha):
        max_size = max([len(x) for x in offspring + population])
        for ind in offspring + population:
            assert alpha >= 0, f"Alpha Value {alpha}"
            setattr(ind, "original_fitness", ind.fitness.values)
            ind.fitness.weights = (-1, -1)
            ind.fitness.values = (
                ind.fitness.values[0],
                len(ind) / max_size + alpha * ind.fitness.values[0],
            )
