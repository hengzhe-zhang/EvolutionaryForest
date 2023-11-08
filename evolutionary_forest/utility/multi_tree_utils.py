import copy
from typing import TYPE_CHECKING

from deap.gp import PrimitiveTree, cxOnePoint

from evolutionary_forest.multigene_gp import (
    MultipleGeneGP,
    genHalfAndHalf_with_prob,
)

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def gene_addition(
    individual: MultipleGeneGP, algorithm: "EvolutionaryForestRegressor", tree=None
):
    if len(individual.gene) < individual.max_gene_num:
        if tree is not None:
            individual.gene.append(tree)
            return

        # not add the same gene
        existing_genes = set([str(g) for g in individual.gene])
        mode = "Random"
        tree = tree_generation(individual, mode, algorithm)
        iteration = 0
        while str(tree) in existing_genes:
            if iteration >= 100:
                # not try to add genes
                return
            tree = PrimitiveTree(individual.content())
            iteration += 1
        individual.gene.append(tree)


def tree_generation(
    individual: MultipleGeneGP, mode, algorithm: "EvolutionaryForestRegressor"
) -> PrimitiveTree:
    if mode == "Crossover":
        gene_a, gene_b = individual.random_select(), individual.random_select()
        tree, _ = cxOnePoint(copy.deepcopy(gene_a), copy.deepcopy(gene_b))
    else:
        if algorithm.estimation_of_distribution.turn_on:
            # probability matching for terminal variables
            min_height, max_height = algorithm.initial_tree_size.split("-")
            min_height, max_height = int(min_height), int(max_height)
            expression = genHalfAndHalf_with_prob(
                algorithm.pset, min_height, max_height, algorithm
            )
            tree = PrimitiveTree(expression)
        else:
            tree = PrimitiveTree(algorithm.toolbox.expr())
    return tree
