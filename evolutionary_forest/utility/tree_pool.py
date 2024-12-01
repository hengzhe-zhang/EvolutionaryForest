import math
from collections import defaultdict
from typing import List, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
from deap.gp import PrimitiveTree, Terminal, MetaEphemeral
from deap.tools import selBest
from scipy.spatial import cKDTree, KDTree
from scipy.special import softmax
from scipy.stats import pearsonr, rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from evolutionary_forest.component.configuration import (
    MAPElitesConfiguration,
    MutationConfiguration,
)
from evolutionary_forest.component.evaluation import single_tree_evaluation
from evolutionary_forest.component.generalization.smoothness import (
    function_second_order_smoothness,
)
from evolutionary_forest.utility.memory.instance_selection import (
    semantic_instance_selection,
)
from evolutionary_forest.utility.mlp_library import (
    NeuralSemanticLibrary,
    filter_train_data_by_node_count,
)
from evolutionary_forest.utility.shapley_tool import copy_and_rename_tree
from evolutionary_forest.utility.tree_pool_util.adaptive_instance_selection import (
    adaptive_selection_strategy_controller,
)
from evolutionary_forest.utility.tree_pool_util.ball_tree import ScipyBallTree

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor

from evolutionary_forest.multigene_gp import MultipleGeneGP, map_elites_selection
from evolutionary_forest.utility.feature_importance_util import (
    feature_importance_process,
)
from evolutionary_forest.utility.normalization_tool import normalize_vector


def sum_rank(first_column, second_column):
    first_column_rank = rankdata(first_column, method="average") - 1
    second_column_rank = rankdata(second_column, method="average") - 1
    # Compute the sum of ranks
    sum_of_ranks = first_column_rank + second_column_rank
    # Get the sorted indices based on the sum of ranks
    sorted_index = np.argsort(sum_of_ranks)
    return sorted_index


def get_irrelevant_features(pearson_matrix, top_features):
    # Step 1: Create a dictionary of feature importances
    importance_dict = {
        feature_id: importance for feature_id, importance in top_features
    }
    # Step 2: Sort features by importance in descending order
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    # Step 3: Initialize correlation data structure
    correlated_variables = {i: set() for i in importance_dict.keys()}
    # Populate correlated_variables based on the Pearson matrix
    for row in importance_dict.keys():
        for col in importance_dict.keys():
            if pearson_matrix[row][col] > 0.99:
                correlated_variables[col].add(row)
                correlated_variables[row].add(col)
    added_list = set()
    forbidden_list = set()
    # Step 5: Process features to update forbidden list
    for feature_id, importance in sorted_features:
        # Check if this feature is correlated with any high-importance feature
        if any(
            correlated_feature in added_list
            for correlated_feature in correlated_variables.get(feature_id, [])
        ):
            forbidden_list.add(f"ARG{feature_id}")
        else:
            added_list.add(feature_id)
    return forbidden_list


def plot_distribution(nearest_samples):
    sns.set(style="whitegrid")
    plt.hist(nearest_samples, edgecolor="black")
    plt.xlabel("Sample")
    plt.ylabel("Frequency")
    plt.title("Distribution of Samples")
    plt.show()


def select_samples_via_quantiles(y: np.ndarray, n_bins=100):
    # Reshape y to a column vector
    y_reshaped = y.reshape(-1, 1)

    # Initialize KBinsDiscretizer
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")

    # Fit KBinsDiscretizer and transform y to clusters
    clusters = kbd.fit_transform(y_reshaped)

    # Convert cluster indices to integer
    cluster_indices = clusters.astype(int).reshape(-1)

    # Initialize an array to hold the index of one sample per cluster
    selected_indexes = []

    # Iterate over unique cluster indices
    for cluster_index in np.unique(cluster_indices):
        # Find indices of samples belonging to the current cluster
        cluster_samples_indices = np.where(cluster_indices == cluster_index)[0]

        # Select the first sample index from the current cluster
        selected_indexes.append(cluster_samples_indices[0])
    return selected_indexes


class SemanticLibrary:
    def __init__(
        self,
        max_trees=1000,
        library_updating_mode="LeastFrequentUsed",
        semantics_length=5,
        random_order_replacement=True,
        pset=None,
        verbose=False,
        mutation_configuration: MutationConfiguration = None,
        x_columns=0,
        skip_single_terminal=False,
        library_implementation="KD-Tree",
        **params,
    ):
        self.library_implementation = library_implementation
        self.plain_semantics_list = []
        self.clustering_indexes = None
        self.frequency = defaultdict(int)
        self.plot_mismatch = False
        self.plot_distance = False
        self.library_updating_mode = library_updating_mode
        self.kd_tree: KDTree = None  # This will be a cKDTree instance
        self.trees = []  # List to store PrimitiveTree objects
        self.normalized_semantics_list = []  # List to store normalized semantics
        self.curiosity = []
        # Set to store hashes of seen semantics for uniqueness
        self.seen_semantics = dict()
        self.seen_semantics_counter = 0
        self.seen_trees = set()
        self.max_trees = max_trees  # Maximum number of trees to store
        # Length of semantics to use for KD-Tree
        self.semantics_length = semantics_length
        self.random_order_replacement = random_order_replacement
        self.verbose = verbose
        self.target_semantics: np.ndarray = None
        self.skip_single_terminal = skip_single_terminal

        self.log_initialization()
        self.forbidden_list = set()
        self.previous_loss_matrix = None
        self.mutation_configuration = mutation_configuration
        if mutation_configuration.neural_pool != 0:
            self.mlp_pool: NeuralSemanticLibrary = NeuralSemanticLibrary(
                input_size=min(semantics_length, x_columns),
                hidden_size=self.mutation_configuration.neural_pool_hidden_size,
                num_layers=self.mutation_configuration.neural_pool_mlp_layers,
                pset=pset,
                output_primitive_length=self.mutation_configuration.neural_pool_num_of_functions,
                transformer_layers=self.mutation_configuration.neural_pool_transformer_layer,
                selective_retrain=self.mutation_configuration.selective_retrain,
                retrieval_augmented_generation=self.mutation_configuration.retrieval_augmented_generation,
                use_decoder_transformer="encoder-decoder",
            )
        else:
            self.mlp_pool = None

    def log_initialization(self):
        self.mismatch_times = []
        self.distance_distribution = []

    def callback(self):
        pass
        if self.verbose:
            pass
            if len(self.frequency) == 0:
                return
            # frequency = np.array(list(self.frequency.values()))
            # print(
            #     "Max Curiosity: ",
            #     # np.array(self.curiosity)[np.argsort(self.curiosity)[-10:]],
            #     np.sum(frequency[np.argsort(frequency)[-10:]]),
            # )

    def forbidden_check(self, tree):
        for node in tree:
            if isinstance(node, Terminal) and node.name in self.forbidden_list:
                return True
            # if isinstance(node, Primitive) and node.name in self.forbidden_list:
            #     return True
        return False

    def append_full_tree(
        self, inds: List[MultipleGeneGP], target_semantics: np.ndarray
    ):
        target_semantics = self.index_semantics(target_semantics)
        normalized_target_semantics = normalize_vector(target_semantics)
        forbidden_counter = 0
        total_counter = 0

        for ind in inds:
            for semantics, tree in zip(ind.semantics.T, ind.gene):
                semantics = self.index_semantics(semantics)
                if isinstance(semantics, torch.Tensor):
                    semantics = semantics.detach().numpy()
                if np.linalg.norm(semantics) == 0:
                    continue  # Skip this semantics as its norm is 0

                normalized_semantics = normalize_vector(semantics)
                if (
                    np.isnan(normalized_semantics).any()
                    or np.isinf(normalized_semantics).any()
                ):
                    continue

                semantics_hash = tuple(normalized_semantics)
                if semantics_hash in self.seen_semantics:
                    continue
                total_counter += 1
                if self.forbidden_check(tree):
                    forbidden_counter += 1
                    continue

                # if str(tree) in self.seen_trees:
                #     raise ValueError("Duplicate tree")

                self.seen_semantics[semantics_hash] = len(tree)
                self.trees.append(tree)
                self.curiosity.append(0)
                # self.seen_trees.add(str(tree))
                # self.plain_semantics_list.append(semantics)
                self.normalized_semantics_list.append(
                    normalized_semantics
                )  # Store the normalized semantics
                if self.mutation_configuration.negative_data_augmentation:
                    self.trees.append(tree)
                    self.normalized_semantics_list.append(-1 * normalized_semantics)
                    self.curiosity.append(0)
                    self.seen_semantics[tuple(-1 * normalized_semantics)] = len(tree)

        self.clean_when_full(normalized_target_semantics)
        if self.verbose:
            print("Forbidden Counter: ", forbidden_counter)
        # print("Number of trees in the library: ", len(self.trees))
        if len(self.normalized_semantics_list) > 0:
            # Create the KDTree with all collected points
            if self.library_implementation == "KD-Tree":
                self.kd_tree = cKDTree(self.normalized_semantics_list)
            elif self.library_implementation == "Ball-Tree":
                self.kd_tree = ScipyBallTree(
                    self.normalized_semantics_list, leaf_size=16
                )
            else:
                raise Exception("Invalid library implementation")
            assert len(self.trees) == len(self.normalized_semantics_list)
        self.frequency.clear()

    def append_subtree(self, semantics: np.ndarray, tree: PrimitiveTree):
        semantics = self.index_semantics(semantics)
        if np.linalg.norm(semantics) == 0:
            return  # Skip this semantics as its norm is 0

        normalized_semantics = normalize_vector(semantics)
        if np.isnan(normalized_semantics).any() or np.isinf(normalized_semantics).any():
            return

        semantics_hash = tuple(normalized_semantics)
        if semantics_hash in self.seen_semantics:
            # self.seen_semantics_counter += 1
            # if self.seen_semantics_counter % 1000 == 0:
            #     print(self.seen_semantics_counter, len(self.trees))
            return

        self.seen_semantics[semantics_hash] = len(tree)
        self.trees.append(tree)
        self.normalized_semantics_list.append(
            normalized_semantics
        )  # Store the normalized semantics
        self.curiosity.append(0)
        if self.mutation_configuration.negative_data_augmentation:
            self.trees.append(tree)
            self.normalized_semantics_list.append(
                (-1 * normalized_semantics)
            )  # Store the normalized semantics
            self.curiosity.append(0)
            self.seen_semantics[tuple(-1 * normalized_semantics)] = len(tree)

    def clean_when_full(self, normalized_target_semantics):
        # Handle excess trees
        if len(self.trees) > self.max_trees:
            assert len(self.trees) == len(self.normalized_semantics_list)
            assert len(self.trees) == len(self.curiosity)
            indexes = np.arange(len(self.trees))
            if self.library_updating_mode == "MAP-Elites":
                indexes = self.update_by_map_elites(
                    self.normalized_semantics_list, normalized_target_semantics
                )
                indexes = np.array(indexes)
            elif self.library_updating_mode == "Recent":
                excess = len(self.trees) - self.max_trees
                indexes = indexes[excess:]
            elif self.library_updating_mode == "LeastFrequentUsed":
                indexes = sorted(
                    indexes, key=lambda x: (self.frequency.get(x, 0), x), reverse=True
                )[: self.max_trees]
                self.frequency.clear()
            elif self.library_updating_mode == "Curiosity":
                indexes = sorted(
                    indexes, key=lambda x: self.curiosity[x], reverse=True
                )[: self.max_trees]
            else:
                raise ValueError("Invalid updating mode")
            # Remove the oldest trees
            self.trees = [self.trees[idx] for idx in indexes]
            self.normalized_semantics_list = [
                self.normalized_semantics_list[idx] for idx in indexes
            ]
            self.curiosity = [self.curiosity[idx] for idx in indexes]
            # self.plain_semantics_list = [
            #     self.plain_semantics_list[idx] for idx in indexes
            # ]
            # Recreate seen_semantics based on the current normalized_semantics_list
            self.seen_semantics = {
                tuple(semantics): len(tree)
                for semantics, tree in zip(self.normalized_semantics_list, self.trees)
            }

    def update_by_map_elites(
        self,
        normalized_semantics_list: List[np.ndarray],
        target_semantics,
    ):
        correlation_to_targets = [
            pearsonr(sem, target_semantics)[0] for sem in normalized_semantics_list
        ]
        kpca = Pipeline(
            [
                ("Standardization", StandardScaler(with_mean=False)),
                ("KPCA", KernelPCA(kernel="cosine", n_components=2)),
            ]
        )

        kpca_semantics = kpca.fit_transform(np.array(normalized_semantics_list))
        map_elites_configuration = MAPElitesConfiguration(map_elites_bins=30)
        idx = map_elites_selection(
            kpca_semantics, correlation_to_targets, math.inf, map_elites_configuration
        )
        return idx

    def retrieve_nearest_tree(
        self, semantics: np.ndarray, return_semantics=False, negative_search=True
    ):
        if self.kd_tree is None:
            raise ValueError("KD-Tree is empty. Please add some trees first.")

        semantics = self.index_semantics(semantics)

        # Normalize the query semantics
        norm = np.linalg.norm(semantics)
        if norm > 0:
            semantics = semantics / norm
        else:
            return None

        # Query the KDTree for the nearest point
        dist, index = self.kd_tree.query(semantics)
        dist_neg, index_neg = self.kd_tree.query(-semantics)
        if negative_search and dist_neg < dist:
            dist = dist_neg
            index = index_neg

        self.plot_distance_function(dist)
        # if self.library_updating_mode == "LeastFrequentUsed":
        self.frequency[index] += 1
        if return_semantics:
            return self.trees[index], self.normalized_semantics_list[index], index
        return self.trees[index]  # Return the corresponding tree

    def retrieve_smallest_nearest_tree(
        self,
        semantics: np.ndarray,
        return_semantics=False,
        top_k=10,
        incumbent_size=math.inf,
        incumbent_depth=math.inf,
        incumbent_distance=math.inf,
        negative_search=True,
        weight_vector=None,
        complexity_function=None,
    ):
        if self.kd_tree is None:
            raise ValueError("KD-Tree is empty. Please add some trees first.")

        if len(self.normalized_semantics_list) == 0:
            # Empty KD-Tree
            return None

        semantics = self.index_semantics(semantics)

        # Normalize the query semantics
        norm = np.linalg.norm(semantics)
        if norm > 0:
            semantics = semantics / norm
        else:
            return None

        # Query the KDTree for the nearest point
        if negative_search:
            # Reduce the query set to half
            dist, index = self.kd_tree.query(semantics, k=top_k // 2)
            dist_neg, index_neg = self.kd_tree.query(-semantics, k=top_k // 2)
            index = np.concatenate([index, index_neg])
            # From short to long
            dist = np.concatenate([dist, dist_neg])
            sorted_index = np.argsort(dist)
        else:
            dist, index = self.kd_tree.query(semantics, k=top_k)
            sorted_index = np.argsort(dist)

        if weight_vector is not None:
            dist = np.array(
                [
                    np.linalg.norm(
                        (semantics - self.normalized_semantics_list[idx])
                        * weight_vector
                    )
                    for idx in index
                ]
            )

        index = index[sorted_index]
        dist = dist[sorted_index]
        # [len(self.trees[index[idx]]) for idx in range(len(index))]
        # for idx in index[sorted_index]:
        #     print(self.trees[idx], self.frequency[idx])

        if incumbent_size == 0:
            smallest_index = np.argmin([len(self.trees[idx]) for idx in index])
            if len(self.trees[smallest_index]) > incumbent_size:
                # If the smallest tree is larger than the incumbent size,
                # return None
                return None
        else:
            # self.check_constant_trees()
            # Find the smallest tree that satisfies the constraints
            smallest_index = -1
            for idx in range(len(index)):
                proposed_tree = self.trees[index[idx]]
                custom_complexity_criteria = complexity_function is not None and (
                    complexity_function(
                        proposed_tree, self.normalized_semantics_list[index[idx]]
                    )
                    <= incumbent_size
                )
                if (
                    (
                        (len(proposed_tree) <= incumbent_size)
                        or custom_complexity_criteria
                    )
                    and proposed_tree.height <= incumbent_depth
                    and dist[idx] < incumbent_distance
                ):
                    if (
                        len(proposed_tree) == 1
                        and isinstance(proposed_tree[0], Terminal)
                        and self.skip_single_terminal
                    ):
                        # Skip tree with single terminal
                        # Empirically lead to faster convergence, smaller tree size
                        continue
                    smallest_index = index[idx]
                    # str(self.trees[index[idx]])
                    break

        if smallest_index == -1:
            # Non suitable tree
            return None

        if return_semantics:
            return (
                self.trees[smallest_index],
                self.normalized_semantics_list[smallest_index],
                smallest_index,
            )
        return self.trees[smallest_index]  # Return the corresponding tree

    def check_constant_trees(self):
        for idx, tree in enumerate(self.trees):
            if tree[0].name == "rand101":
                print(idx, tree, self.normalized_semantics_list[idx])

    def retrieve_smooth_nearest_tree(
        self,
        semantics: np.ndarray,
        return_semantics=False,
        top_k=10,
        incumbent_smooth=math.inf,
        negative_search=True,
        smoothness_function=function_second_order_smoothness,
        best_one=False,
        focus_one_target=False,
    ):
        if self.kd_tree is None:
            raise ValueError("KD-Tree is empty. Please add some trees first.")

        if len(self.normalized_semantics_list) == 0:
            # Empty KD-Tree
            return None

        semantics = self.index_semantics(semantics)

        # Normalize the query semantics
        norm = np.linalg.norm(semantics)
        if norm > 0:
            semantics = semantics / norm
        else:
            return None

        # Query the KDTree for the nearest point
        dist, index = self.kd_tree.query(semantics, k=top_k)
        if negative_search:
            dist_neg, index_neg = self.kd_tree.query(-semantics, k=top_k)
            index = np.concatenate([index, index_neg])
            # From short to long
            sorted_index = np.argsort(np.concatenate([dist, dist_neg]))
        else:
            sorted_index = np.argsort(dist)
        index = index[sorted_index][:top_k]

        reference = semantics
        if focus_one_target:
            reference = self.target_semantics[self.clustering_indexes]
        if best_one:
            # choose the minimum one
            smallest_index = np.argmin(
                [
                    smoothness_function(
                        self.normalized_semantics_list[index[idx]],
                        reference,
                    )
                    for idx in range(top_k)
                ]
            )
            return (
                self.trees[smallest_index],
                self.normalized_semantics_list[smallest_index],
            )

        smallest_index = -1
        for idx in range(top_k):
            if (
                smoothness_function(
                    self.normalized_semantics_list[index[idx]],
                    reference,
                )
                <= incumbent_smooth
            ):
                smallest_index = index[idx]
                break
        # for idx in range(10):
        #     finding = self.normalized_semantics_list[index[idx]]
        #     target = normalize_vector(self.target_semantics[self.clustering_indexes])
        #     idxs = np.argsort(semantics)
        #     plt.plot(mean_of_parts(finding[idxs], 5))
        #     plt.plot(mean_of_parts(target[idxs], 5))
        #     plt.plot(mean_of_parts(semantics[idxs], 5))
        #     plt.legend(["Finding", "Target", "Query"])
        #     plt.title(
        #         f"Smoothness: {idx}, Tree Size:{len(self.trees[index[idx]])}, Smoothness: {smoothness_function(finding, semantics)}\n Tree:{str(self.trees[index[idx]])}"
        #     )
        #     plt.show()

        # for idx in range(top_k):
        #     print(idx, self.trees[index[idx]])

        if smallest_index == -1:
            return None

        # if self.library_updating_mode == "LeastFrequentUsed":
        self.frequency[smallest_index] += 1

        if return_semantics:
            return (
                self.trees[smallest_index],
                self.normalized_semantics_list[smallest_index],
            )
        return self.trees[smallest_index]  # Return the corresponding tree

    def index_semantics(self, semantics):
        if self.clustering_indexes is None or len(semantics) <= len(
            self.clustering_indexes
        ):
            return semantics
        if self.clustering_indexes is None:
            semantics = semantics[: self.semantics_length]
        else:
            semantics = semantics[self.clustering_indexes]
        return semantics

    def retrieve_nearest_trees_weighted(self, semantics: np.ndarray, top_k=10, std=1):
        if self.kd_tree is None:
            raise ValueError("KD-Tree is empty. Please add some trees first.")

        semantics = self.index_semantics(semantics)

        # Normalize the query semantics
        norm = np.linalg.norm(semantics)
        if norm > 0:
            semantics = semantics / norm
        else:
            return None

        # Query the KDTree for the nearest points
        distances, indices = self.kd_tree.query(semantics, k=top_k)

        # Generate random weight vector
        weight_vector = np.random.normal(loc=0, scale=std, size=semantics.shape)
        weight_vector = softmax(weight_vector)

        # Initialize variables for storing the best match
        best_distance = np.inf
        best_tree_index = None

        # Iterate over the top-k trees and compare based on weighted Euclidean distance
        for dist, index in zip(distances, indices):
            tree_semantics = self.normalized_semantics_list[index]
            weighted_euclidean_distance = np.sqrt(
                np.sum(weight_vector * (semantics - tree_semantics) ** 2)
            )
            if weighted_euclidean_distance < best_distance:
                best_distance = weighted_euclidean_distance
                best_tree_index = index
        if self.plot_mismatch:
            # Determine the index with the smallest unweighted distance for comparison
            unweighted_best_index = indices[np.argmin(distances)]

            # Check if the best weighted index is different from the best unweighted index
            if best_tree_index != unweighted_best_index:
                self.mismatch_times.append(1)
        self.plot_distance_function(best_distance)
        if self.library_updating_mode == "LeastFrequentUsed":
            self.frequency[best_tree_index] += 1
        return self.trees[best_tree_index]  # Return the corresponding tree

    def plot_distance_function(self, best_distance):
        if self.plot_distance:
            self.distance_distribution.append(best_distance)
            if (
                len(self.distance_distribution) > 0
                and len(self.distance_distribution) % 200 == 0
            ):
                plt.hist(self.distance_distribution, bins=10, color="blue", alpha=0.7)
                plt.title("Distance Distribution")
                plt.xlabel("Distance")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
                self.distance_distribution.clear()

    def update_forbidden_list(
        self,
        pop,
        pearson_matrix,
        feature_selection_mode,
        fs_mode,
        algorithm: "EvolutionaryForestRegressor",
    ):
        # self.group_selection(pop, total_features)
        # Count feature usage in each group
        features = defaultdict(float)
        # fs_mode = "Frequency"
        if fs_mode == "Shapley":
            pop = sorted(pop, key=lambda x: x.fitness.wvalues[0])[:20]
        for ind in pop:
            # Currently only supporting Linear Regression with LOOCV
            assert np.allclose(
                feature_importance_process(abs(ind.pipe["Ridge"].coef_)), ind.coef
            )
            for coef, tree in zip(ind.coef, ind.gene):
                if fs_mode == "Frequency":
                    for node in tree:
                        if isinstance(node, Terminal) and node.name.startswith("ARG"):
                            feature_id = int(node.name.replace("ARG", ""))
                            features[feature_id] += coef * ind.fitness.wvalues[0]
                        # if isinstance(node, Primitive):
                        #     features[node.name] += coef * ind.fitness.wvalues[0]
                else:
                    tree_copy, used_features, mapping_dict = copy_and_rename_tree(tree)
                    if len(used_features) == 0:
                        continue
                    evaluation_function = lambda data: single_tree_evaluation(
                        tree_copy, algorithm.pset, data=data
                    )
                    data = algorithm.X[:3]
                    data = data[:, used_features]
                    data = StandardScaler().fit_transform(data)
                    explainer = shap.SamplingExplainer(
                        model=evaluation_function, data=data
                    )

                    # Calculate Shapley values
                    shap_values = explainer.shap_values(data, silent=True)
                    average_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
                    inverse_mapping_dict = {v: k for k, v in mapping_dict.items()}

                    for i, feature_name in enumerate(mapping_dict.values()):
                        original_feature_name = inverse_mapping_dict[feature_name]
                        feature_id = int(original_feature_name.replace("ARG", ""))
                        features[feature_id] += coef * average_abs_shap_values[i]
        # Top cumulative sum 95% features
        total_importance = sum(features.values())
        cumulative_sum = 0
        top_features = []

        for feature_id, importance in sorted(
            features.items(), key=lambda x: x[1], reverse=True
        ):
            cumulative_sum += importance
            top_features.append((feature_id, importance))
            threshold = feature_selection_mode
            if cumulative_sum / total_importance >= threshold:
                break

        top_feature_ids = set([feature_id for feature_id, importance in top_features])

        forbidden_list = get_irrelevant_features(pearson_matrix, top_features)
        self.forbidden_list.update(forbidden_list)

        for f_id in features.keys():
            if f_id not in top_feature_ids:
                self.forbidden_list.add(f"ARG{f_id}")

        # for f_id in range(total_features):
        #     if f_id not in top_feature_ids:
        #         self.forbidden_list.add(f"ARG{f_id}")
        # print("Number of features: ", len(features.keys()))
        # for primitive in [p.name for p in algorithm.pset.primitives[object]]:
        #     if primitive not in top_feature_ids:
        #         self.forbidden_list.add(primitive)

        if self.verbose:
            # Print the top 95% features
            print(f"Top {int(feature_selection_mode * 100)}% features:")
            for feature_id, importance in top_features:
                if f"ARG{feature_id}" not in forbidden_list:
                    print(f"Feature ARG{feature_id}: Importance {importance}")

    def group_selection(self, pop, total_features):
        # Calculate the size of each feature group
        features_per_group = total_features // 3
        feature_group = {0: 0, 1: 0, 2: 0}
        # Select the top 30 individuals
        top_individuals = selBest(pop, 30)
        # Count feature usage in each group
        for ind in top_individuals:
            # Currently only supporting Linear Regression with LOOCV
            assert np.allclose(
                feature_importance_process(abs(ind.pipe["Ridge"].coef_)), ind.coef
            )
            for coef, tree in zip(ind.coef, ind.gene):
                for node in tree:
                    if isinstance(node, Terminal) and node.name.startswith("ARG"):
                        feature_id = int(node.name.replace("ARG", ""))
                        group_id = feature_id // features_per_group
                        if group_id in feature_group:
                            feature_group[group_id] += coef * ind.fitness.wvalues[0]

            # if self.verbose:
            #     for ind in sorted(pop, key=lambda x: x.fitness.wvalues[0]):
            #         print(str(ind), ind.fitness.wvalues[0])
        # Find the largest feature group
        largest_group = max(feature_group, key=feature_group.get)
        if self.verbose:
            print("Largest group: ", largest_group, "Feature group: ", feature_group)
        # Ban features from other groups
        for f_id in range(total_features):
            if f_id // features_per_group != largest_group:
                self.forbidden_list.add(f"ARG{f_id}")

    def update_hard_instance(
        self,
        error: np.ndarray,
        semantics: np.ndarray,
        mode: str,
        features: np.ndarray,
        label: np.ndarray,
        current_generation,
        total_generations,
    ):
        if len(error.T) <= self.semantics_length:
            # no need to do clustering
            return None
        if mode == "K-Means":
            k_means = KMeans(n_clusters=self.semantics_length)
            k_means.fit_transform(error.T)
            centroids = k_means.cluster_centers_  # Get centroids of each cluster

            # Find the index of the nearest sample to each centroid
            nearest_indexes, _ = pairwise_distances_argmin_min(centroids, error.T)
            self.clustering_indexes = nearest_indexes
        elif mode == "Semantic-K-Means":
            k_means = KMeans(n_clusters=self.semantics_length)
            k_means.fit_transform(semantics.T)
            centroids = k_means.cluster_centers_  # Get centroids of each cluster
            # Find the index of the nearest sample to each centroid
            nearest_indexes, _ = pairwise_distances_argmin_min(centroids, semantics.T)
            self.clustering_indexes = nearest_indexes
        elif mode == "Label-K-Means":
            k_means = KMeans(n_clusters=self.semantics_length)
            k_means.fit_transform(label.reshape(-1, 1))
            centroids = k_means.cluster_centers_  # Get centroids of each cluster
            # Find the index of the nearest sample to each centroid
            nearest_indexes, _ = pairwise_distances_argmin_min(
                centroids, label.reshape(-1, 1)
            )
            self.clustering_indexes = nearest_indexes
        elif mode == "Feature-K-Means":
            k_means = KMeans(n_clusters=self.semantics_length)
            k_means.fit_transform(features)
            centroids = k_means.cluster_centers_  # Get centroids of each cluster
            # Find the index of the nearest sample to each centroid
            nearest_indexes, _ = pairwise_distances_argmin_min(centroids, features)
            self.clustering_indexes = nearest_indexes
        elif mode == "Hardest-KMeans":
            self.clustering_indexes = semantic_instance_selection(
                error, self.semantics_length
            )
        elif mode == "Worst":
            errors = np.median(error, axis=0)
            self.clustering_indexes = np.argsort(errors)[-self.semantics_length :]
        elif mode == "Easy":
            errors = np.median(error, axis=0)
            self.clustering_indexes = np.argsort(errors)[: self.semantics_length]
        elif mode == "Random":
            self.clustering_indexes = np.random.choice(
                np.arange(error.shape[1]), self.semantics_length, replace=False
            )
        elif mode.startswith("adaptive") or mode.startswith("curriculum"):
            self.clustering_indexes = adaptive_selection_strategy_controller(
                mode,
                error,
                np.arange(error.shape[1]),
                self.previous_loss_matrix,
                self.semantics_length,
                current_generation,
                total_generations,
            )
            self.previous_loss_matrix = error
        else:
            raise Exception("Invalid mode")
        self.clustering_indexes = self.clustering_indexes[
            np.argsort(label[self.clustering_indexes])
        ]
        self.clear_all()

    def clear_all(self):
        self.frequency.clear()
        self.curiosity.clear()
        self.trees.clear()
        self.normalized_semantics_list.clear()
        self.plain_semantics_list.clear()
        self.seen_trees.clear()
        self.seen_semantics.clear()
        self.kd_tree = None
        self.log_initialization()

    def set_clustering_based_semantics(self, y, mode: str):
        if len(y) <= self.semantics_length:
            # no need to do clustering
            return None
        if mode == "Quantiles":
            nearest_indexes = select_samples_via_quantiles(
                y, n_bins=self.semantics_length
            )
            self.clustering_indexes = nearest_indexes
            # plot_distribution(y[nearest_indexes])
            # plot_distribution(y)
        elif mode == "K-Means":
            k_means = KMeans(n_clusters=self.semantics_length)
            k_means.fit_transform(y.reshape(-1, 1))
            centroids = k_means.cluster_centers_  # Get centroids of each cluster

            # Find the index of the nearest sample to each centroid
            nearest_indexes, _ = pairwise_distances_argmin_min(
                centroids, y.reshape(-1, 1)
            )
            self.clustering_indexes = nearest_indexes
        else:
            raise ValueError("Invalid mode")

    def plot_top_frequencies(self, top_n=10):
        # Get the top N most frequently accessed trees
        top_frequencies = sorted(
            self.frequency.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        top_indices = [str(self.trees[index]) for index, freq in top_frequencies]
        top_values = [freq for index, freq in top_frequencies]
        # Check if both lists have the same length
        if len(top_indices) != len(top_values):
            print("Mismatch in lengths of top_indices and top_values")
            return

        # Ensure there are elements to plot
        if len(top_indices) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(15, 6))
        # Print the top frequencies
        print("Top Frequencies:")
        for i, freq in enumerate(top_values):
            print(f"Tree {top_indices[i]}: {freq} times")

        # Plot the top frequencies
        plt.barh(range(len(top_indices)), top_values, tick_label=top_indices)
        plt.xlabel("Frequency")
        plt.ylabel("Tree Index")
        plt.title("Top Frequencies of Retrieved Trees")
        plt.tight_layout()
        plt.show()

    def train_nn(self):
        if self.mutation_configuration.neural_pool != 0:
            self.mlp_pool: NeuralSemanticLibrary
            train_data = [
                (tree, semantics)
                for tree, semantics in zip(self.trees, self.normalized_semantics_list)
            ]
            if self.mutation_configuration.negative_data_augmentation:
                assert len(train_data) % 2 == 0, f"Train Data: {len(train_data)}"
                assert list(self.trees[0]) == list(
                    self.trees[1]
                ), f"Tree 0: {self.trees[0]}, Tree 1: {self.trees[1]}"
                train_data = [
                    (tree, semantics)
                    for index, (tree, semantics) in enumerate(train_data)
                    if index % 2 == 1
                ]
                assert (
                    len(train_data) == len(self.trees) // 2
                ), f"Train Data: {len(train_data)}, Trees: {len(self.trees)}"
            train_data = filter_train_data_by_node_count(
                train_data, max_function_nodes=self.mlp_pool.output_primitive_length
            )
            self.mlp_pool.train(
                train_data,
                lr=0.01,
                verbose=False,
                patience=5,
                loss_weight=self.mutation_configuration.weight_of_contrastive_learning,
            )
