import itertools
import math
import random

from deap.tools import sortNondominated, selBest


def ceil_to_even(number):
    # Calculate the ceiling of the number
    ceil_value = math.ceil(number)

    # If the ceiling value is even, return it
    if ceil_value % 2 == 0:
        return ceil_value
    else:
        # If the ceiling value is odd, return the next even number
        return ceil_value + 1


def sel_subset_best(pop, k, subset_ratio=0.1):
    # Number of models to randomly select in each subset
    subset_size = int(len(pop) * subset_ratio)
    # Container for the selected breeding stock
    breeding_stock = []

    while len(breeding_stock) < k:
        # Randomly select a subset of models
        subset = random.sample(pop, subset_size)
        breeding_stock.append(selBest(subset, 1)[0])
    return breeding_stock


def sel_pareto_tournament(pop, k, subset_size):
    # Container for the selected breeding stock
    breeding_stock = []

    while len(breeding_stock) < k:
        # Randomly select a subset of models
        subset = random.sample(pop, min(subset_size, len(pop)))

        # Identify the Pareto front of the subset
        pareto_fronts = sortNondominated(subset, len(subset), first_front_only=True)
        pareto_front = pareto_fronts[0]

        # Add all models on the Pareto front to the breeding stock
        breeding_stock.extend(pareto_front)

    return breeding_stock[: len(breeding_stock) // 2 * 2]
    # return random.sample(breeding_stock, k)


def pareto_tournament_controller(offspring, version="Product"):
    if version == "Product":
        combinations = list(itertools.combinations(offspring, 2))
        offspring = [item for sublist in combinations for item in sublist]
    elif version == "Adjacent":
        offspring = [
            [offspring[i], offspring[i + 1]] for i in range(len(offspring) - 1)
        ]
        offspring = [item for sublist in offspring for item in sublist]
    elif version == "Consecution":
        offspring = [
            [offspring[i], offspring[i + 1]] for i in range(0, len(offspring) - 1, 2)
        ]
        offspring = [item for sublist in offspring for item in sublist]
    else:
        raise Exception("Invalid version")
    return offspring
