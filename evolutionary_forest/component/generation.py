import random
from typing import List

import numpy as np
from deap.gp import mutUniform
from deap.tools import cxTwoPoint
from scipy.stats import pearsonr, spearmanr

from evolutionary_forest.component.configuration import CrossoverConfiguration, MutationConfiguration
from evolutionary_forest.component.evaluation import quick_evaluate
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.multigene_gp import gene_crossover, MultipleGeneGP, gene_mutation


def varAndPlus(population, toolbox: TypedToolbox, cxpb, mutpb, gene_num, limitation_check,
               semantic_check_tool=None,
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
                random.random() < crossover_configuration.macro_crossover_rate:
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
                    if random.random() < mutpb and (not crossover_configuration.var_or
                                                    or (i not in crossed_individual)):
                        offspring[i], = toolbox.mutate(offspring[i])
                        # if crossover already modifies an individual,
                        # then set its parent fitness as the fitness values of two parents
                        if offspring[i].parent_fitness is None:
                            parent_fitness = offspring[i].fitness.wvalues[0]
                            offspring[i].parent_fitness = (parent_fitness,)

                for c in range(invokes):
                    if random.random() < mutation_configuration.gene_addition_rate:
                        offspring[i].gene_addition()

                    if random.random() < mutation_configuration.gene_deletion_rate:
                        offspring[i].gene_deletion()
            del offspring[i].fitness.values

            if semantic_check_tool is not None:
                # check by semantics
                x = semantic_check_tool['x']
                y = semantic_check_tool['y']
                pset = semantic_check_tool['pset']
                correlation_threshold = semantic_check_tool.get('correlation_threshold', 0.2)
                correlation_mode = semantic_check_tool.get('correlation_mode', 'Pearson')
                index = np.random.randint(0, len(x), 20)
                for k, g in enumerate(offspring[i].gene):
                    y_hat = quick_evaluate(g, pset, x[index])
                    c = 0
                    function = {
                        'Pearson': pearsonr,
                        'Spearman': spearmanr,
                    }[correlation_mode]
                    while (not isinstance(y_hat, np.ndarray)) or (y_hat.size != y[index].size) or \
                        (np.abs(function(y_hat, y[index])[0]) < correlation_threshold):
                        c += 1
                        offspring[i].gene[k] = mutUniform(g, toolbox.expr_mut, pset)[0]
                        del offspring[i].fitness.values
                        y_hat = quick_evaluate(g, pset, x[index])
                        if c > 100:
                            print('Warning!')
                            break
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
