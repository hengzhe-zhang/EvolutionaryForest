import copy
import math
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
from evolutionary_forest.component.external_archive.semantic_library_mode_controller import (
    semantic_library_mode_controller,
)
from evolutionary_forest.component.generalization.smoothness import (
    function_first_order_smoothness,
)
from evolutionary_forest.component.gradient_optimization.linear_scaling import (
    calculate_slope,
    calculate_intercept,
)
from evolutionary_forest.component.stgp.strongly_type_gp_utility import revert_back
from evolutionary_forest.component.toolbox import TypedToolbox
from evolutionary_forest.utility.deletion_utils import *
from evolutionary_forest.utility.normalization_tool import normalize_vector

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
        #     # remove redundant
        #     for x in offspring:
        #         ele = set()
        #         indicies = []
        #         for k, v in enumerate(x.hash_result):
        #             if v not in ele:
        #                 ele.add(v)
        #                 indicies.append(k)
        #         x.gene: list = [x.gene[i] for i in indicies]
        if not mutation_configuration.addition_or_deletion:
            for i in range(len(offspring)):
                addition_and_deletion(i, offspring)
        if mutation_configuration.pool_based_addition:
            # exclusive
            for i in range(len(offspring)):
                tree_replacement(offspring[i])
            algorithm.tree_pool.callback()
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
                invokes = get_number_of_invokes(offspring[i].gene_num)
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
            # new_trees = random.randint(
            #     1, offspring[i].max_gene_num - len(offspring[i].gene) + 1
            # )
            # for _ in range(new_trees):
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

    def tree_replacement(ind: MultipleGeneGP):
        indexes = algorithm.tree_pool.clustering_indexes
        if indexes is None:
            indexes = list(range(len(ind.individual_semantics)))
        current_semantics = ind.individual_semantics[indexes]
        target = algorithm.y[indexes]

        orders = list(range(len(ind.gene)))
        if algorithm.tree_pool.random_order_replacement:
            random.shuffle(orders)

        mutation_configuration = algorithm.mutation_configuration
        # prediction_validation(ind, indexes)

        for id in orders[: mutation_configuration.local_search_step]:
            delete_semantics = ind.pipe["Ridge"].coef_[id] * (
                (ind.semantics[indexes, id] - ind.scaler.mean_[id])
                / np.where(ind.scaler.scale_[id] == 0, 1, ind.scaler.scale_[id])
            )
            temp_semantics = current_semantics - delete_semantics

            if (
                random.random()
                > mutation_configuration.pool_based_replacement_inner_probability
            ):
                return

            if mutation_configuration.scaling_before_replacement:
                factor = calculate_slope(temp_semantics, target)
                intercept = calculate_intercept(temp_semantics, target, factor)
                temp_semantics = temp_semantics * factor + intercept

            residual = target - temp_semantics

            if mutation_configuration.complementary_replacement > 0:
                ratio = mutation_configuration.complementary_replacement
                residual = (1 - ratio) * target + ratio * (target - temp_semantics)

            if algorithm.verbose:
                algorithm.success_rate.add_values(0)

            pool_addition_mode = mutation_configuration.pool_addition_mode
            pool_addition_mode = semantic_library_mode_controller(
                pool_addition_mode,
                current_gen=algorithm.current_gen,
                n_gen=algorithm.n_gen,
            )
            if pool_addition_mode.startswith("Smooth"):
                if pool_addition_mode == "Smooth-First":
                    smoothness_function = function_first_order_smoothness
                else:
                    raise Exception
                if pool_addition_mode == "Smooth-Target":
                    incumbent_smooth = smoothness_function(
                        normalize_vector(delete_semantics), normalize_vector(target)
                    )
                else:
                    incumbent_smooth = smoothness_function(
                        normalize_vector(delete_semantics), normalize_vector(residual)
                    )
                value = algorithm.tree_pool.retrieve_smooth_nearest_tree(
                    normalize_vector(residual),
                    return_semantics=True,
                    incumbent_size=len(ind.gene[id]),
                    incumbent_smooth=incumbent_smooth,
                    top_k=mutation_configuration.top_k_candidates,
                    negative_search=mutation_configuration.negative_local_search,
                    smoothness_function=smoothness_function,
                    focus_one_target=pool_addition_mode == "Smooth-Target",
                )
            elif (
                pool_addition_mode == "Smallest"
                or pool_addition_mode.startswith("Smallest~Auto")
                or pool_addition_mode.startswith("Smallest~Curiosity")
            ):
                incumbent_depth = math.inf
                if pool_addition_mode == "Smallest~Auto":
                    incumbent_size = len(ind.gene[id])
                elif pool_addition_mode == "Smallest~Auto-Depth":
                    incumbent_depth = ind.gene[id].height
                    incumbent_size = math.inf
                elif mutation_configuration.pool_addition_mode.startswith(
                    "Smallest~Curiosity"
                ):
                    incumbent_size = len(ind.gene[id])
                    if "Depth" in mutation_configuration.pool_addition_mode:
                        incumbent_size = math.inf
                        incumbent_depth = ind.gene[id].height
                else:
                    incumbent_size = 0
                # str(ind.gene[id])
                if pool_addition_mode.startswith("Smallest~CuriosityIS"):
                    curiosity_driven = float(pool_addition_mode.split("-")[-1])
                    if algorithm.current_gen < curiosity_driven * algorithm.n_gen:
                        # explore first
                        curiosity_driven = 1
                    else:
                        # then exploit
                        curiosity_driven = -1
                elif pool_addition_mode.startswith("Smallest~CuriosityS"):
                    curiosity_driven = float(pool_addition_mode.split("-")[-1])
                    if algorithm.current_gen < curiosity_driven * algorithm.n_gen:
                        # exploit first
                        curiosity_driven = -1
                    else:
                        # then explore
                        curiosity_driven = 1
                elif pool_addition_mode.startswith("Smallest~Curiosity"):
                    curiosity_driven = -1
                else:
                    curiosity_driven = 0
                value = algorithm.tree_pool.retrieve_smallest_nearest_tree(
                    normalize_vector(residual),
                    return_semantics=True,
                    incumbent_size=incumbent_size,
                    incumbent_depth=incumbent_depth,
                    top_k=mutation_configuration.top_k_candidates,
                    negative_search=mutation_configuration.negative_local_search,
                    curiosity_driven=curiosity_driven,
                )
            else:
                value = algorithm.tree_pool.retrieve_nearest_tree(
                    normalize_vector(residual),
                    return_semantics=True,
                    negative_search=mutation_configuration.negative_local_search,
                )

            if value is None:
                continue
            tree, proposed_semantics = value

            if np.all(
                normalize_vector(ind.semantics[indexes, id]) == proposed_semantics
            ):
                continue
            if mutation_configuration.full_scaling_after_replacement:
                lr = LinearRegression()
                lr.fit(np.vstack((temp_semantics, proposed_semantics)).T, target)
                trail_semantics = lr.predict(
                    np.vstack((temp_semantics, proposed_semantics)).T
                )
            else:
                factor = calculate_slope(proposed_semantics, residual)
                intercept = calculate_intercept(proposed_semantics, residual, factor)
                trail_semantics = (
                    temp_semantics + factor * proposed_semantics + intercept
                )

            trial_mse = np.mean((trail_semantics - target) ** 2)
            current_mse = np.mean((current_semantics - target) ** 2)
            if trial_mse <= mutation_configuration.trial_check_ratio * current_mse or (
                not mutation_configuration.trial_check
            ):
                # replacement
                ind.gene[id] = copy.deepcopy(tree)
                current_semantics = trail_semantics
                if algorithm.verbose:
                    algorithm.success_rate.add_values(1)
                    if algorithm.success_rate.get_total_count() % 1000 == 0:
                        print(
                            "Success Rate",
                            algorithm.success_rate.get_moving_averages(),
                            algorithm.success_rate.get_total_count(),
                        )
            else:
                if (
                    algorithm.mutation_configuration.lib_feature_selection
                    and algorithm.tree_pool.forbidden_check(ind.gene[id])
                ):
                    value = algorithm.tree_pool.retrieve_nearest_tree(
                        normalize_vector(residual),
                        return_semantics=True,
                        negative_search=mutation_configuration.negative_local_search,
                    )
                    tree, proposed_semantics = value
                    factor = calculate_slope(proposed_semantics, residual)
                    intercept = calculate_intercept(
                        proposed_semantics, residual, factor
                    )
                    trail_semantics = (
                        temp_semantics + factor * proposed_semantics + intercept
                    )
                    trial_mse = np.mean((trail_semantics - target) ** 2)
                    if trial_mse <= current_mse:
                        ind.gene[id] = copy.deepcopy(tree)
                        current_semantics = trail_semantics
                        algorithm.fs_success_rate.add_values(1)
                    else:
                        algorithm.fs_success_rate.add_values(0)

                    # print(
                    #     "Current Gen",
                    #     current_gen,
                    #     "Success Rate",
                    #     algorithm.fs_success_rate.get_moving_averages(),
                    # )
                else:
                    pass
                pass

        return

    def prediction_validation(ind, indexes):
        semantics = np.zeros_like(indexes, dtype=np.float32)
        for id in range(0, len(ind.gene)):
            semantics += ind.pipe["Ridge"].coef_[id] * (
                (ind.semantics[indexes, id] - ind.scaler.mean_[id])
                / np.where(ind.scaler.scale_[id] == 0, 1, ind.scaler.scale_[id])
            )
        semantics += ind.pipe["Ridge"].intercept_

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
