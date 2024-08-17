import numpy as np

from evolutionary_forest.component.selection_operators.non_dominated_sorting import (
    sortNondominatedCustom,
    AutoWeightFitness,
)


def sel_lexicase_pareto_tournament_weighted_subset(pop, k, subset_size):
    weights = np.median([ind.case_values for ind in pop], axis=0)
    subset = np.random.choice(
        len(pop[0].case_values), subset_size, replace=False, p=weights
    )
    offspring = []
    while len(offspring) < 2:
        pareto_fronts = sortNondominatedCustom(
            pop,
            len(pop),
            lambda ind: AutoWeightFitness(tuple(ind.case_values[subset])),
            first_front_only=True,
        )
        offspring.extend(pareto_fronts[0])
    offspring = offspring[: len(offspring) // 2 * 2]
    return offspring


def sel_lexicase_pareto_tournament_random_subset(pop, k, subset_size):
    subset = np.random.choice(len(pop[0].case_values), subset_size, replace=False)
    offspring = []
    while len(offspring) < 2:
        pareto_fronts = sortNondominatedCustom(
            pop,
            len(pop),
            lambda ind: AutoWeightFitness(tuple(ind.case_values[subset])),
            first_front_only=True,
        )
        offspring.extend(pareto_fronts[0])
    offspring = offspring[: len(offspring) // 2 * 2]
    return offspring
