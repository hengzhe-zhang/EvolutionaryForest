import copy
import random
from typing import TYPE_CHECKING

from deap.gp import PrimitiveTree, cxOnePoint

from evolutionary_forest.multigene_gp import (
    MultipleGeneGP,
    genHalfAndHalf_with_prob,
)
from evolutionary_forest.utility.normalization_tool import normalize_vector

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def random_replacement(
    individual: MultipleGeneGP, algorithm: "EvolutionaryForestRegressor", tree=None
):
    for i, gene in enumerate(individual.gene):
        if random.random() < algorithm.mutation_configuration.gene_replacement_rate:
            tree = tree_generation(individual, "", algorithm)
            individual.gene[i] = tree
    return individual


def gene_addition(
    individual: MultipleGeneGP, algorithm: "EvolutionaryForestRegressor", tree=None
):
    if len(individual.gene) < individual.max_gene_num:
        if tree is not None:
            individual.gene.append(tree)
            return

        # not add the same gene
        existing_genes = set([str(g) for g in individual.gene])
        mutation_configuration = algorithm.mutation_configuration
        gene_addition_mode = mutation_configuration.gene_addition_mode
        if (
            mutation_configuration.pool_based_addition
            and algorithm.tree_pool.kd_tree is not None
        ):
            residual = algorithm.y - individual.individual_semantics
            residual = normalize_vector(residual)
            if mutation_configuration.pool_addition_mode == "Best":
                tree = copy.deepcopy(
                    algorithm.tree_pool.retrieve_nearest_tree(residual)
                )
            else:
                raise ValueError("Invalid pool addition mode")
        else:
            tree = tree_generation(individual, gene_addition_mode, algorithm)

        if tree is None:
            return

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
    if mode == "Crossover":
        # crossover with the randomly generated tree to preserve diversity
        tree, _ = cxOnePoint(tree, individual.random_select())
    return tree
