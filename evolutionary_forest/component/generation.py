import copy
import math
import random
from typing import List, TYPE_CHECKING

import numpy as np
from deap.tools import cxTwoPoint

from evolutionary_forest.component.configuration import (
    CrossoverConfiguration,
    MutationConfiguration,
)
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.utility.deletion_utils import find_most_redundant_feature

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


from evolutionary_forest.multigene_gp import (
    gene_crossover,
    MultipleGeneGP,
    gene_mutation,
)
from evolutionary_forest.utility.multi_tree_utils import gene_addition


def varAndPlus(
    population,
    toolbox: TypedToolbox,
    cxpb,
    mutpb,
    limitation_check,
    crossover_configuration: CrossoverConfiguration = None,
    mutation_configuration: MutationConfiguration = None,
    algorithm: "EvolutionaryForestRegressor" = None,
):
    if crossover_configuration is None:
        crossover_configuration = CrossoverConfiguration()
    if mutation_configuration is None:
        mutation_configuration = MutationConfiguration()

    @limitation_check
    def mutation_function(*population):
        if mutation_configuration.pool_based_addition:
            for ind in population:
                ind.individual_semantics = ind.predicted_values
        offspring: List[MultipleGeneGP] = [toolbox.clone(ind) for ind in population]

        if crossover_configuration.var_or:
            # only execute crossover *or* mutation
            return varOr(offspring)
        # if mutation_configuration.addition_or_deletion:
        #     # remove redundant
        #     for x in offspring:
        #         ele = set()
        #         indicies = []
        #         for k, v in enumerate(x.hash_result):
        #             if v not in ele:
        #                 ele.add(v)
        #                 indicies.append(k)
        #         x.gene: list = [x.gene[i] for i in indicies]

        # Apply crossover and mutation on the offspring
        # Support both VarAnd and VarOr
        i = 0
        while i < len(offspring):
            # in default, gene_num is the number of trees in each individual
            gene_num = offspring[i].gene_num

            # Execute mutation and selection operator N-times
            if (
                i % 2 == 0
                and crossover_configuration.macro_crossover_rate > 0
                and random.random() < crossover_configuration.macro_crossover_rate
                and min(len(offspring[i].gene), len(offspring[i + 1].gene)) > 1
            ):
                genetic_algorithm_style_macro_crossover(offspring, i)
                if crossover_configuration.independent_macro_crossover:
                    # skip micro-crossover after macro-crossover
                    i += 2
                    continue

            if isinstance(cxpb, np.ndarray):
                # MGP crossover
                if crossover_configuration.number_of_invokes > 0:
                    invokes = crossover_configuration.number_of_invokes
                else:
                    # iterate through all trees
                    invokes = offspring[i].gene_num
                for c in range(invokes):
                    assert not crossover_configuration.var_or, "Not supported varOr!"
                    if i % 2 == 0 and random.random() < cxpb[c]:
                        offspring[i].gene[c], offspring[i + 1].gene[c] = gene_crossover(
                            offspring[i].gene[c],
                            offspring[i + 1].gene[c],
                            configuration=crossover_configuration,
                        )

                    if random.random() < mutpb[c]:
                        (offspring[i].gene[c],) = gene_mutation(
                            offspring[i].gene[c],
                            toolbox.pset.pset_list[c],
                            toolbox.expr_mut,
                            toolbox.tree_generation,
                            mutation_configuration,
                        )
            else:
                if i % 2 == 0:
                    if offspring[i].crossover_type != "Macro":
                        # only reset if not in macro-crossover state
                        offspring[i].parent_fitness = None
                    if offspring[i + 1].crossover_type != "Macro":
                        # only reset if not in macro-crossover state
                        offspring[i + 1].parent_fitness = None
                    # for crossover, the number of genes the minimum of two
                    gene_num = min(offspring[i].gene_num, offspring[i + 1].gene_num)

                # crossover, using the smallest number of genes for a pair of individuals
                invokes = get_number_of_invokes(gene_num)
                for c in range(invokes):
                    if i % 2 == 0 and random.random() < cxpb:
                        offspring[i], offspring[i + 1] = toolbox.mate(
                            offspring[i], offspring[i + 1]
                        )

                        # set parent fitness as the fitness values of two parents
                        if offspring[i].parent_fitness is None:
                            offspring[i].parent_fitness = (
                                offspring[i].fitness.wvalues[0],
                                offspring[i + 1].fitness.wvalues[0],
                            )
                        if offspring[i + 1].parent_fitness is None:
                            offspring[i + 1].parent_fitness = (
                                offspring[i + 1].fitness.wvalues[0],
                                offspring[i].fitness.wvalues[0],
                            )

                # mutation, using the number of genes for each individual
                invokes = get_number_of_invokes(offspring[i].gene_num)
                for c in range(invokes):
                    # If in var_or mode, once crossed, not allowed to be mutated
                    if random.random() < mutpb:
                        (offspring[i],) = toolbox.mutate(offspring[i])
                        # if crossover already modifies an individual,
                        # then set its parent fitness as the fitness values of two parents
                        if offspring[i].parent_fitness is None:
                            parent_fitness = offspring[i].fitness.wvalues[0]
                            offspring[i].parent_fitness = (parent_fitness,)

                if mutation_configuration.addition_or_deletion:
                    addition_or_deletion(i, offspring)
                else:
                    addition_and_deletion(i, offspring)
            del offspring[i].fitness.values
            i += 1

        for o in offspring:
            # must delete all fitness values
            assert (
                not hasattr(o, "fitness")
                or not hasattr(o.fitness, "values")
                or len(o.fitness.values) == 0
            )
        return offspring

    def genetic_algorithm_style_macro_crossover(offspring, i):
        # two point crossover means crossover operators in GA, not in GP
        offspring[i].gene, offspring[i + 1].gene = cxTwoPoint(
            offspring[i].gene, offspring[i + 1].gene
        )
        del offspring[i].fitness.values
        del offspring[i + 1].fitness.values

    def addition_or_deletion(i, offspring):
        addition_and_deletion = random.random()
        if addition_and_deletion < mutation_configuration.gene_addition_rate:
            gene_addition(offspring[i], algorithm)
        elif (
            addition_and_deletion
            < mutation_configuration.gene_addition_rate
            + mutation_configuration.gene_deletion_rate
        ):
            if mutation_configuration.redundant_based_deletion:
                if len(offspring[i].gene) > 1:
                    worst_one = find_most_redundant_feature(offspring[i].semantics)
                    del offspring[i].gene[worst_one]
            elif mutation_configuration.weighted_deletion:
                offspring[i].gene_deletion(weighted=True)
            else:
                offspring[i].gene_deletion()

    def addition_and_deletion(i, offspring):
        if random.random() < mutation_configuration.gene_addition_rate:
            gene_addition(offspring[i], algorithm)
        if random.random() < mutation_configuration.gene_deletion_rate:
            offspring[i].gene_deletion()

    def varOr(offspring):
        # Allocate indexes for genetic operators
        for i in range(0, len(offspring), 2):
            r = random.random()
            if r < mutpb:
                # mutation
                for k in range(i, i + 2):
                    rr = random.random()
                    if rr < mutation_configuration.gene_addition_rate:
                        gene_addition(offspring[k], algorithm)
                        del offspring[k].fitness.values
                    elif rr < mutation_configuration.gene_deletion_rate:
                        offspring[k].gene_deletion()
                        del offspring[k].fitness.values
                    else:
                        (offspring[k],) = toolbox.mutate(offspring[k])
                        del offspring[k].fitness.values
            elif r < cxpb + mutpb:
                # crossover
                (
                    offspring[i],
                    offspring[i + 1],
                ) = toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
            else:
                # reproduction
                for k in range(i, i + 2):
                    offspring[k] = [copy.deepcopy(offspring[k])]
                    del offspring[k].fitness.values
        return offspring

    def get_number_of_invokes(gene_num):
        if crossover_configuration.number_of_invokes > 0:
            invokes = crossover_configuration.number_of_invokes
            if invokes < 1:
                # based on ratio
                invokes = math.ceil(gene_num * invokes)
        else:
            invokes = gene_num
        return invokes

    return mutation_function(*population)
