import numpy as np
from deap.tools import sortNondominated


def fitness_normalization(
    individuals,
    classification_task,
    non_dominated_normalization=False,
    classification_loss="CrossEntropy",
):
    for ind in individuals:
        ind.unnormalized_fitness = ind.fitness.values

    if non_dominated_normalization:
        non_dominated = sortNondominated(
            individuals, len(individuals), first_front_only=True
        )[0]
        if len(non_dominated) > 1:
            min_max_pop = non_dominated
        else:
            min_max_pop = individuals
    else:
        min_max_pop = individuals

    dims = len(individuals[0].fitness.values)
    min_max = []
    for d in range(dims):
        values = [ind.fitness.values[d] for ind in min_max_pop]
        min_val = min(values)
        max_val = max(values)
        min_max.append((min_val, max_val))
    for ind in individuals:
        values = []
        for d in range(dims):
            fitness_value = ind.fitness.values[d]
            if classification_task and classification_loss == "CrossEntropy":
                # for cross-entropy, need exponential normalization
                if d == 0:
                    """
                    Simply take exponential of the fitness value to transform cross entropy to linear scale.
                    """
                    fitness_value = np.exp(ind.fitness.values[d])
            min_val, max_val = min_max[d]
            normalized_fitness = fitness_value - min_val
            if (max_val - min_val) > 0:
                normalized_fitness = normalized_fitness / (max_val - min_val)
            values.append(normalized_fitness)
        ind.fitness.values = values


def fitness_restore_back(individuals):
    for ind in individuals:
        ind.fitness.values = ind.unnormalized_fitness
