import copy
import random
from typing import List, TYPE_CHECKING

from deap.tools import cxTwoPoint
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from evolutionary_forest.component.bloat_control.simple_simplification import (
    simple_simplification,
)
from evolutionary_forest.component.configuration import (
    CrossoverConfiguration,
    MutationConfiguration,
)
from evolutionary_forest.component.mutation.tree_replacement import tree_replacement
from evolutionary_forest.component.stgp.strongly_type_gp_utility import revert_back
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.component.variation.variation_util import get_number_of_invokes
from evolutionary_forest.utility.deletion_utils import *

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


from evolutionary_forest.multigene_gp import (
    gene_crossover,
    MultipleGeneGP,
    gene_mutation,
)
from evolutionary_forest.utility.multi_tree_utils import (
    gene_addition,
    random_replacement,
)


def remove_semantically_redundant_tree(offspring):
    # remove redundant
    for x in offspring:
        x: MultipleGeneGP
        ele = set()
        indicies = []
        for k, v in enumerate(x.hash_result):
            if v not in ele:
                ele.add(v)
                indicies.append(k)
        x.gene: list = [x.gene[i] for i in indicies]


def pool_mode_controller(pool_addition_mode, X, y):
    cv_score = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")
    et_cv_score = cross_val_score(ExtraTreesRegressor(), X, y, cv=5, scoring="r2")
    if pool_addition_mode == "Adaptive":
        if cv_score.mean() <= 0.4 and np.mean(et_cv_score) >= 0.8:
            pool_addition_mode = "Best"
        else:
            pool_addition_mode = "Smallest"
    return pool_addition_mode


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
                ind.individual_semantics = ind.pipe.predict(ind.semantics)
                # ind.individual_semantics = ind.predicted_values
                ind.scaler = ind.pipe.named_steps["Scaler"]
        offspring: List[MultipleGeneGP] = [toolbox.clone(ind) for ind in population]

        if crossover_configuration.var_or:
            # only execute crossover *or* mutation
            return varOr(offspring)
        # if mutation_configuration.addition_or_deletion:
        #     remove_semantically_redundant_tree(offspring)
        if not mutation_configuration.addition_or_deletion:
            for i in range(len(offspring)):
                addition_and_deletion(i, offspring)
        if mutation_configuration.pool_based_addition:
            if mutation_configuration.semantic_local_search_pb == "Adaptive":
                (
                    selection_operator,
                    selection_operator_id,
                ) = algorithm.aos.one_step_sample()
                local_search_flag = selection_operator == "LocalSearch"
                for o in offspring:
                    o.selection_operator = selection_operator_id
                # print(f"Selection Operator: {selection_operator}")
            else:
                local_search_flag = (
                    random.random() < mutation_configuration.semantic_local_search_pb
                )

            if local_search_flag:
                # exclusive
                for i in range(len(offspring)):
                    tree_replacement(offspring[i], algorithm)

                for o in offspring:
                    o.generation_operator = "LocalSearch"

                algorithm.tree_pool.callback()
                if (mutation_configuration.semantic_local_search_pb != 1) and (
                    mutation_configuration.independent_local_search is None
                    or mutation_configuration.independent_local_search is True
                ):
                    # independent mode
                    return offspring
            else:
                for o in offspring:
                    o.generation_operator = "Evolution"
        # Apply crossover and mutation on the offspring
        # Support both VarAnd and VarOr
        i = 0
        while i < len(offspring):
            # in default, gene_num is the number of trees in each individual
            gene_num = offspring[i].gene_num

            if i % 2 == 0 and crossover_configuration.tree_selection == "PseudoRandom":
                offspring[i].pseudo_random.shuffle()
                offspring[i + 1].pseudo_random.shuffle()

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
                invokes = get_number_of_invokes(gene_num, crossover_configuration)
                for c in range(invokes):
                    if i % 2 == 0 and random.random() < cxpb:
                        offspring[i], offspring[i + 1] = toolbox.mate(
                            offspring[i], offspring[i + 1]
                        )

                        # set parent fitness as the fitness values of two parents
                        if (
                            offspring[i].parent_fitness is None
                            and len(offspring[i].fitness.wvalues) > 0
                        ):
                            offspring[i].parent_fitness = (
                                offspring[i].fitness.wvalues[0],
                                offspring[i + 1].fitness.wvalues[0],
                            )
                        if (
                            offspring[i + 1].parent_fitness is None
                            and len(offspring[i + 1].fitness.wvalues) > 0
                        ):
                            offspring[i + 1].parent_fitness = (
                                offspring[i + 1].fitness.wvalues[0],
                                offspring[i].fitness.wvalues[0],
                            )

                # mutation, using the number of genes for each individual
                invokes = get_number_of_invokes(
                    offspring[i].gene_num, crossover_configuration
                )
                for c in range(invokes):
                    # If in var_or mode, once crossed, not allowed to be mutated
                    if random.random() < mutpb:
                        (offspring[i],) = toolbox.mutate(offspring[i])
                        # if crossover already modifies an individual,
                        # then set its parent fitness as the fitness values of two parents
                        if (
                            offspring[i].parent_fitness is None
                            and len(offspring[i].fitness.wvalues) > 0
                        ):
                            parent_fitness = offspring[i].fitness.wvalues[0]
                            offspring[i].parent_fitness = (parent_fitness,)

                if mutation_configuration.addition_or_deletion:
                    addition_or_deletion(i, offspring)
                    if mutation_configuration.handle_objective_duplication:
                        simple_simplification(offspring[i])
                # else:
                #     addition_and_deletion(i, offspring)
            del offspring[i].fitness.values
            i += 1

        for o in offspring:
            if algorithm.mutation_configuration.gene_replacement_rate > 0:
                random_replacement(o, algorithm)
            # must delete all fitness values
            assert (
                not hasattr(o, "fitness")
                or not hasattr(o.fitness, "values")
                or len(o.fitness.values) == 0
            )
            if mutation_configuration.basic_primitives.startswith("Pipeline"):
                revert_back(o)
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
        assert (
            mutation_configuration.gene_addition_rate
            + mutation_configuration.gene_deletion_rate
            <= 1
        )
        if addition_and_deletion < mutation_configuration.gene_addition_rate:
            gene_addition(offspring[i], algorithm)
        elif (
            addition_and_deletion
            < mutation_configuration.gene_addition_rate
            + mutation_configuration.gene_deletion_rate
        ):
            if mutation_configuration.redundant_based_deletion:
                if len(offspring[i].gene) > 1:
                    if mutation_configuration.deletion_strategy == "Redundancy":
                        worst_one = find_most_redundant_feature(offspring[i].semantics)
                    elif mutation_configuration.deletion_strategy == "Redundancy+":
                        worst_one = find_most_redundant_feature(
                            offspring[i].semantics, inverse=True
                        )
                    elif mutation_configuration.deletion_strategy == "Importance":
                        worst_one = np.argmin(abs(offspring[i].coef))
                    elif mutation_configuration.deletion_strategy == "Importance+":
                        worst_one = np.argmax(abs(offspring[i].coef))
                    elif mutation_configuration.deletion_strategy == "Backward":
                        worst_one = find_least_useful_feature(
                            offspring[i].semantics, algorithm.y
                        )
                    elif mutation_configuration.deletion_strategy == "Backward+":
                        worst_one = find_least_useful_feature(
                            offspring[i].semantics, algorithm.y, inverse=True
                        )
                    elif mutation_configuration.deletion_strategy == "P-Value":
                        worst_one = find_largest_p_value_feature(
                            offspring[i].semantics, algorithm.y
                        )
                    elif mutation_configuration.deletion_strategy == "P-Value+":
                        worst_one = find_largest_p_value_feature(
                            offspring[i].semantics, algorithm.y, inverse=True
                        )
                    elif mutation_configuration.deletion_strategy == "Correlation":
                        worst_one = find_least_correlation_to_target(
                            offspring[i].semantics, algorithm.y
                        )
                    elif mutation_configuration.deletion_strategy == "Correlation+":
                        worst_one = find_least_correlation_to_target(
                            offspring[i].semantics, algorithm.y, inverse=True
                        )
                    else:
                        raise ValueError(
                            f"Unknown deletion strategy: {mutation_configuration.deletion_strategy}"
                        )
                    del offspring[i].gene[worst_one]
            elif mutation_configuration.weighted_deletion:
                offspring[i].gene_deletion(weighted=True)
            else:
                offspring[i].gene_deletion()

    def addition_and_deletion(i, offspring):
        if random.random() < mutation_configuration.gene_deletion_rate:
            random_index = offspring[i].gene_deletion()
            if (
                random_index is not None
                and mutation_configuration.pool_based_addition
                and mutation_configuration.change_semantic_after_deletion
            ):
                # This is important to ensure correct residual
                offspring[i].individual_semantics -= (
                    offspring[i].coef[random_index]
                    * offspring[i].semantics[:, random_index]
                )
        if random.random() < mutation_configuration.gene_addition_rate:
            gene_addition(offspring[i], algorithm)

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

    return mutation_function(*population)
