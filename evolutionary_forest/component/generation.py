import random
from typing import List

import numpy as np
from deap.tools import cxTwoPoint

from evolutionary_forest.component.configuration import CrossoverConfiguration, MutationConfiguration
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.multigene_gp import gene_crossover, MultipleGeneGP, gene_mutation


def varAndPlus(population, toolbox: TypedToolbox, cxpb, mutpb, limitation_check,
               crossover_configuration: CrossoverConfiguration = None,
               mutation_configuration: MutationConfiguration = None):
    if crossover_configuration is None:
        crossover_configuration = CrossoverConfiguration()
    if mutation_configuration is None:
        mutation_configuration = MutationConfiguration()

    @limitation_check
    def mutation_function(*population):
        offspring: List[MultipleGeneGP] = [toolbox.clone(ind) for ind in population]
        crossed_individual = set()

        # Apply crossover and mutation on the offspring
        # Support both VarAnd and VarOr
        i = 0
        while i < len(offspring):
            # Execute mutation and selection operator N-times
            if i % 2 == 0 and crossover_configuration.macro_crossover_rate > 0 and \
                random.random() < crossover_configuration.macro_crossover_rate and \
                min(len(offspring[i].gene), len(offspring[i + 1].gene)) > 1:
                offspring[i].gene, offspring[i + 1].gene = cxTwoPoint(offspring[i].gene, offspring[i + 1].gene)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
                if crossover_configuration.independent_macro_crossover:
                    # skip micro-crossover after macro-crossover
                    i += 2
                    continue

            if isinstance(cxpb, np.ndarray):
                if crossover_configuration.number_of_invokes > 0:
                    invokes = crossover_configuration.number_of_invokes
                else:
                    invokes = gene_num
                for c in range(invokes):
                    assert not crossover_configuration.var_or, "Not supported varOr!"
                    if i % 2 == 0 and random.random() < cxpb[c]:
                        offspring[i].gene[c], offspring[i + 1].gene[c] = gene_crossover(offspring[i].gene[c],
                                                                                        offspring[i + 1].gene[c],
                                                                                        configuration=crossover_configuration)

                    if random.random() < mutpb[c]:
                        offspring[i].gene[c], = gene_mutation(offspring[i].gene[c], toolbox.pset.pset_list[c],
                                                              toolbox.expr_mut, toolbox.tree_generation,
                                                              mutation_configuration)
            else:
                if i % 2 == 0:
                    if offspring[i].crossover_type != 'Macro':
                        # only reset if not in macro-crossover state
                        offspring[i].parent_fitness = None
                    if offspring[i + 1].crossover_type != 'Macro':
                        offspring[i + 1].parent_fitness = None

                    gene_num = min(offspring[i].gene_num, offspring[i + 1].gene_num)

                # crossover, using the smallest number of genes for a pair of individuals
                invokes = get_number_of_invokes(gene_num)
                for c in range(invokes):
                    if i % 2 == 0 and random.random() < cxpb:
                        offspring[i], offspring[i + 1] = toolbox.mate(offspring[i], offspring[i + 1])
                        # del offspring[i].fitness.values, offspring[i + 1].fitness.values
                        crossed_individual.add(offspring[i])
                        crossed_individual.add(offspring[i + 1])

                        # set parent fitness as the fitness values of two parents
                        if offspring[i].parent_fitness is None:
                            offspring[i].parent_fitness = (offspring[i].fitness.wvalues[0],
                                                           offspring[i + 1].fitness.wvalues[0])
                        if offspring[i + 1].parent_fitness is None:
                            offspring[i + 1].parent_fitness = (offspring[i + 1].fitness.wvalues[0],
                                                               offspring[i].fitness.wvalues[0])

                # mutation, using the number of genes for each individual
                gene_num = get_number_of_invokes(offspring[i].gene_num)
                for c in range(invokes):
                    # If in var_or mode, once crossed, not allowed to be mutated
                    if random.random() < mutpb and (not crossover_configuration.var_or
                                                    or (i not in crossed_individual)):
                        offspring[i], = toolbox.mutate(offspring[i])
                        # if crossover already modifies an individual,
                        # then set its parent fitness as the fitness values of two parents
                        if offspring[i].parent_fitness is None:
                            parent_fitness = offspring[i].fitness.wvalues[0]
                            offspring[i].parent_fitness = (parent_fitness,)

                if random.random() < mutation_configuration.gene_addition_rate:
                    offspring[i].gene_addition()

                if random.random() < mutation_configuration.gene_deletion_rate:
                    if mutation_configuration.weighted_deletion:
                        offspring[i].gene_deletion(weighted=True)
                    else:
                        offspring[i].gene_deletion()
            del offspring[i].fitness.values
            i += 1
        return offspring

    def get_number_of_invokes(gene_num):
        if crossover_configuration.number_of_invokes > 0:
            invokes = crossover_configuration.number_of_invokes
            if invokes < 1:
                invokes = int(gene_num * invokes)
        else:
            invokes = gene_num
        return invokes

    return mutation_function(*population)
