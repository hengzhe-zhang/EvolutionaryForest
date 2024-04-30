import numpy as np
from deap.base import Fitness
from deap.tools import HallOfFame


class Individual:
    def __init__(self, predicted_values):
        self.predicted_values = predicted_values
        Fitness.weights = (-1.0,)
        self.fitness = Fitness((0,))


def generate_random_individuals():
    y_target = np.array([1.0, 1.5, 2.0])

    def generate_random_individual(dim):
        return Individual(np.random.rand(dim))

    population_size = 10
    population = [
        generate_random_individual(len(y_target)) for _ in range(population_size)
    ]
    return population, y_target
