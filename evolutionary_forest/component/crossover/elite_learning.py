import copy
import random
from typing import TYPE_CHECKING

from evolutionary_forest.multigene_gp import (
    MultipleGeneGP,
    tree_crossover,
    tree_mutation,
)

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def learn_from_elite_multitree(
    model: "EvolutionaryForestRegressor",
    individuals,
    population,
    s1_probability=0.5,
    elite_ratio=0.2,
):
    """
    Elite learning for multi-tree GP
    Each tree learns independently from elite population

    Args:
        individuals: Single individual or list of individuals to process
        population: Current population
        s1_probability: Probability of S1 (crossover with best) vs S4 (mutate elite)
        elite_ratio: Ratio of population to consider as elite
        toolbox: DEAP toolbox with genetic operators

    Returns:
        Single offspring or list of offspring (matches input type)
    """
    toolbox = model.toolbox
    # Handle both single individual and list of individuals
    if not isinstance(individuals, list):
        individuals = [individuals]
        return_single = True
    else:
        return_single = False

    # Sort population once
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    n_elite = int(elite_ratio * len(population))
    elite = sorted_pop[:n_elite]

    # Process each individual
    offspring_list = []

    for individual in individuals:
        individual: MultipleGeneGP
        n_trees = individual.gene_num
        offspring_trees = []

        for tree_idx in range(n_trees):
            if random.random() < s1_probability:
                # S1: Crossover with best individual's corresponding tree
                elite_ind = elite[0]
                child_tree, _ = tree_crossover(
                    copy.deepcopy(individual.gene[tree_idx]),
                    copy.deepcopy(elite_ind.gene[tree_idx]),
                    configuration=model.crossover_configuration,
                    pset=model.pset,
                )
            else:
                # S4: Mutate elite's corresponding tree
                elite_ind = random.choice(elite)
                child_tree = copy.deepcopy(elite_ind.gene[tree_idx])
                (child_tree,) = tree_mutation(
                    child_tree,
                    model.pset,
                    model.toolbox.expr,
                    model.toolbox.tree_generation,
                    model.mutation_configuration,
                )

            offspring_trees.append(child_tree)

        # Create new multi-tree individual
        offspring = toolbox.clone(individual)
        offspring.gene = offspring_trees
        offspring_list.append(offspring)

    # Return single offspring if input was single individual, otherwise return list
    if return_single:
        return offspring_list[0]
    else:
        return offspring_list
