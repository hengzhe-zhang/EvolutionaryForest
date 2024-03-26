import numpy as np


def fitness_normalization(individuals, classification_task):
    for ind in individuals:
        ind.unnormalized_fitness = ind.fitness.values

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
            fitness_value = ind.fitness.values[d]
            if classification_task:
                # for cross-entropy, need exponential normalization
                if d == 0:
                    """
                    Original problem: Improve prediction probability
                    Minimization Problem: Negative 1 * Probability
                    """
                    fitness_value = -1 * np.mean(np.exp(-ind.case_values))
            min_val, max_val = min_max[d]
            normalized_fitness = fitness_value - min_val
            if (max_val - min_val) > 0:
                normalized_fitness = normalized_fitness / (max_val - min_val)
            values.append(normalized_fitness)
        ind.fitness.values = values


def fitness_restore_back(individuals):
    for ind in individuals:
        ind.fitness.values = ind.unnormalized_fitness
