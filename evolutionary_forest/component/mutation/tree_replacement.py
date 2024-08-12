import copy
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from deap.gp import Terminal, PrimitiveTree
from scipy.special import softmax
from sklearn.linear_model import LinearRegression

from evolutionary_forest.component.external_archive.semantic_library_mode_controller import (
    semantic_library_mode_controller,
)
from evolutionary_forest.component.gradient_optimization.linear_scaling import (
    calculate_slope,
    calculate_intercept,
)
from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utility.normalization_tool import normalize_vector

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def tree_replacement(ind: MultipleGeneGP, algorithm: "EvolutionaryForestRegressor"):
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
    skip_id = set()

    for sorted_idx, id in enumerate(orders):
        if id in skip_id and not mutation_configuration.local_search_dropout_ensemble:
            continue

        incumbent_size = len(ind.gene[id])
        delete_semantics = ind.pipe["Ridge"].coef_[id] * (
            (ind.semantics[indexes, id] - ind.scaler.mean_[id])
            / np.where(ind.scaler.scale_[id] == 0, 1, ind.scaler.scale_[id])
        )
        dropout_trigger_flag = (
            algorithm.current_gen
            >= algorithm.n_gen * mutation_configuration.local_search_dropout_trigger
        )
        drop_probability_flag = (
            random.random() < mutation_configuration.local_search_dropout
            and sorted_idx + 1 < len(orders)
        )
        dropout_mode = dropout_trigger_flag and drop_probability_flag
        if mutation_configuration.local_search_dropout_ensemble:
            just_one_delete_semantics = delete_semantics
        if dropout_mode:
            new_id = orders[sorted_idx + 1]
            delete_semantics += ind.pipe["Ridge"].coef_[new_id] * (
                (ind.semantics[indexes, new_id] - ind.scaler.mean_[new_id])
                / np.where(ind.scaler.scale_[new_id] == 0, 1, ind.scaler.scale_[new_id])
            )
            if mutation_configuration.local_search_dropout_bloat:
                incumbent_size += len(ind.gene[new_id])
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
        if (
            pool_addition_mode == "Smallest"
            or pool_addition_mode.startswith("Smallest~Auto")
            or pool_addition_mode.startswith("Smallest~Curiosity")
        ):
            incumbent_depth = math.inf
            multi_generation_curiosity = True
            curiosity_driven = False
            negative_distance = False
            negative_curiosity = False
            lexicase_sort = False
            incumbent_distance = np.linalg.norm(
                normalize_vector(residual) - normalize_vector(delete_semantics)
            )
            weight_vector = None
            if pool_addition_mode == "Smallest~Auto":
                pass
            elif pool_addition_mode == "Smallest~Auto-Depth":
                incumbent_depth = ind.gene[id].height
                incumbent_size = math.inf
            elif mutation_configuration.pool_addition_mode.startswith(
                "Smallest~Curiosity"
            ):
                curiosity_driven = True
                if pool_addition_mode.startswith("Smallest~CuriosityS"):
                    multi_generation_curiosity = False
                if "Depth+" in pool_addition_mode:
                    plus_depth = int(pool_addition_mode.split("+")[-1])
                    incumbent_size = math.inf
                    incumbent_depth = ind.gene[id].height + plus_depth
                if "Size+" in pool_addition_mode:
                    plus_size = int(pool_addition_mode.split("+")[-1])
                    incumbent_size = incumbent_size + plus_size
                    incumbent_depth = math.inf
                if "Depth~" in pool_addition_mode:
                    plus_depth = int(pool_addition_mode.split("~")[-1])
                    incumbent_size = math.inf
                    incumbent_depth = max(ind.gene[id].height, plus_depth)
                if "Size~" in pool_addition_mode:
                    plus_size = int(pool_addition_mode.split("~")[-1])
                    incumbent_size = max(len(ind.gene[id]), plus_size)
                    incumbent_depth = math.inf
                if "LastHalf" in pool_addition_mode:
                    # Only To Explore in Later Generations
                    if algorithm.current_gen < algorithm.n_gen // 2:
                        curiosity_driven = False
                if "FirstHalf" in pool_addition_mode:
                    # Only To Explore in Early Generations
                    if algorithm.current_gen > algorithm.n_gen // 2:
                        curiosity_driven = False
                if "Probability" in pool_addition_mode:
                    enable_prob = 1
                    for param in pool_addition_mode.split("-"):
                        if "Probability" in param:
                            enable_prob = float(param.split("+")[-1])
                            break
                    if random.random() > enable_prob:
                        # with probability to disable
                        curiosity_driven = False
                if "NegDis" in pool_addition_mode:
                    negative_distance = True
                if "NegCur" in pool_addition_mode:
                    negative_curiosity = True
                if "Lex" in pool_addition_mode:
                    lexicase_sort = True
            else:
                incumbent_size = 0

            value = algorithm.tree_pool.retrieve_smallest_nearest_tree(
                normalize_vector(residual),
                return_semantics=True,
                incumbent_size=incumbent_size,
                incumbent_depth=incumbent_depth,
                incumbent_distance=incumbent_distance,
                top_k=mutation_configuration.top_k_candidates,
                negative_search=mutation_configuration.negative_local_search,
                # curiosity_driven
                curiosity_driven=curiosity_driven,
                multi_generation_curiosity=multi_generation_curiosity,
                negative_distance=negative_distance,
                negative_curiosity=negative_curiosity,
                lexicase_sort=lexicase_sort,
                weight_vector=weight_vector,
            )
        else:
            value = algorithm.tree_pool.retrieve_nearest_tree(
                normalize_vector(residual),
                return_semantics=True,
                negative_search=mutation_configuration.negative_local_search,
            )

        if value is None:
            continue
        tree, proposed_semantics, proposed_index = value

        if np.all(normalize_vector(ind.semantics[indexes, id]) == proposed_semantics):
            continue

        if dropout_mode and mutation_configuration.local_search_dropout_ensemble:
            temp_semantics = current_semantics - just_one_delete_semantics
            residual = target - temp_semantics

        if mutation_configuration.full_scaling_after_replacement:
            lr = LinearRegression()
            lr.fit(np.vstack((temp_semantics, proposed_semantics)).T, target)
            trail_semantics = lr.predict(
                np.vstack((temp_semantics, proposed_semantics)).T
            )
        else:
            factor = calculate_slope(proposed_semantics, residual)
            intercept = calculate_intercept(proposed_semantics, residual, factor)
            trail_semantics = temp_semantics + factor * proposed_semantics + intercept

        trial_mse = np.mean((trail_semantics - target) ** 2)
        current_mse = np.mean((current_semantics - target) ** 2)
        if pool_addition_mode.startswith("Smallest~Curriculum"):
            trial_mse = np.sum((trail_semantics - target) ** 2 * weight_vector)
            current_mse = np.sum((current_semantics - target) ** 2 * weight_vector)
        if trial_mse <= mutation_configuration.trial_check_ratio * current_mse or (
            not mutation_configuration.trial_check
        ):
            # replacement
            ind.gene[id] = copy.deepcopy(tree)
            current_semantics = trail_semantics
            algorithm.tree_pool.frequency[proposed_index] += 1
            algorithm.tree_pool.curiosity[proposed_index] += 1
            if (
                dropout_mode
                and not mutation_configuration.local_search_dropout_ensemble
            ):
                skip_id.add(new_id)
                ind.gene[new_id] = PrimitiveTree([Terminal(0, False, object)])
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
                intercept = calculate_intercept(proposed_semantics, residual, factor)
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

    if len(skip_id) > 0 and not mutation_configuration.local_search_dropout_ensemble:
        useful_features = [
            gene
            for gene in ind.gene
            if not (
                isinstance(gene[0], Terminal)
                and isinstance(gene[0].value, (float, int))
            )
        ]
        if len(ind.gene) >= 1:
            ind.gene = useful_features
    return
