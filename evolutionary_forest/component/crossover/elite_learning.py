import random

from deap import creator


def learn_from_elite_multitree(
    individual, population, s1_probability=0.5, elite_ratio=0.2, toolbox=None
):
    """
    Elite learning for multi-tree GP
    Each tree learns independently from elite population
    """
    n_trees = individual.gene_num()
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    n_elite = int(elite_ratio * len(population))
    elite = sorted_pop[:n_elite]

    # Create offspring with same number of trees
    offspring_trees = []

    for tree_idx in range(n_trees):
        if random.random() < s1_probability:
            # S1: Crossover with best individual's corresponding tree
            elite_ind = elite[0]

            child_tree, _ = toolbox.mate(
                toolbox.clone(individual[tree_idx]), toolbox.clone(elite_ind[tree_idx])
            )
        else:
            # S4: Mutate elite's corresponding tree
            elite_ind = random.choice(elite)
            child_tree = toolbox.clone(elite_ind[tree_idx])
            (child_tree,) = toolbox.mutate(child_tree)

        offspring_trees.append(child_tree)

    # Create new multi-tree individual
    offspring = creator.Individual(offspring_trees)

    return offspring
