from deap.tools import sortNondominated
import random


def sel_pareto_tournament(pop, k, subset_ratio=0.1):
    # Number of models to randomly select in each subset
    subset_size = int(len(pop) * subset_ratio)

    # Container for the selected breeding stock
    breeding_stock = []

    while len(breeding_stock) < k:
        # Randomly select a subset of models
        subset = random.sample(pop, subset_size)

        # Identify the Pareto front of the subset
        pareto_fronts = sortNondominated(subset, len(subset), first_front_only=True)
        pareto_front = pareto_fronts[0]

        # Add all models on the Pareto front to the breeding stock
        breeding_stock.extend(pareto_front)

        # Ensure we do not exceed the required population size
        if len(breeding_stock) > k:
            random.shuffle(breeding_stock)
            breeding_stock = breeding_stock[:k]

    return breeding_stock
