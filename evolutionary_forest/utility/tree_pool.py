import math
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from deap.gp import PrimitiveTree, Terminal
from deap.tools import selBest
from scipy.spatial import cKDTree, KDTree
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from evolutionary_forest.component.configuration import MAPElitesConfiguration
from evolutionary_forest.multigene_gp import MultipleGeneGP, map_elites_selection
from evolutionary_forest.utility.feature_importance_util import (
    feature_importance_process,
)
from evolutionary_forest.utility.normalization_tool import normalize_vector


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
        verbose=False,
        **params,
    ):
        self.plain_semantics_list = []
        self.clustering_indexes = None
        self.frequency = defaultdict(int)
        self.plot_mismatch = False
        self.plot_distance = False
        self.library_updating_mode = library_updating_mode
        self.kd_tree: KDTree = None  # This will be a cKDTree instance
        self.trees = []  # List to store PrimitiveTree objects
        self.normalized_semantics_list = []  # List to store normalized semantics
        # Set to store hashes of seen semantics for uniqueness
        self.seen_semantics = dict()
        self.seen_semantics_counter = 0
        self.seen_trees = set()
        self.max_trees = max_trees  # Maximum number of trees to store
        # Length of semantics to use for KD-Tree
        self.semantics_length = semantics_length
        self.random_order_replacement = random_order_replacement
        self.verbose = verbose

        self.log_initialization()
        self.forbidden_list = []

    def log_initialization(self):
        self.mismatch_times = []
        self.distance_distribution = []

    def forbidden_check(self, tree):
        for terminal_name in self.forbidden_list:
            for node in tree:
                if isinstance(node, Terminal) and node.name == terminal_name:
                    return True
        return False

    def update_kd_tree(self, inds: List[MultipleGeneGP], target_semantics: np.ndarray):
        target_semantics = self.index_semantics(target_semantics)
        normalized_target_semantics = normalize_vector(target_semantics)

        for ind in inds:
            for semantics, tree in zip(ind.semantics.T, ind.gene):
                semantics = self.index_semantics(semantics)
                # Normalize semantics and skip if norm is 0
                norm = np.linalg.norm(semantics)
                if norm == 0:
                    continue  # Skip this semantics as its norm is 0

                normalized_semantics = normalize_vector(semantics)
                if (
                    np.isnan(normalized_semantics).any()
                    or np.isinf(normalized_semantics).any()
                ):
                    continue

                semantics_hash = tuple(normalized_semantics)
                if (
                    semantics_hash in self.seen_semantics
                    and len(tree) > self.seen_semantics[semantics_hash]
                ):
                    continue
                if self.forbidden_check(tree):
                    continue

                # if str(tree) in self.seen_trees:
                #     raise ValueError("Duplicate tree")

                self.seen_semantics[semantics_hash] = len(tree)
                self.trees.append(tree)
                # self.seen_trees.add(str(tree))
                # self.plain_semantics_list.append(semantics)
                self.normalized_semantics_list.append(
                    normalized_semantics
                )  # Store the normalized semantics

        self.clean_when_full(normalized_target_semantics)
        # print("Number of trees in the library: ", len(self.trees))
        # Create the KDTree with all collected points
        self.kd_tree = cKDTree(self.normalized_semantics_list)

    def append_semantics(self, semantics: np.ndarray, tree: PrimitiveTree):
        semantics = self.index_semantics(semantics)
        # Normalize semantics and skip if norm is 0
        norm = np.linalg.norm(semantics)
        if norm == 0:
            return  # Skip this semantics as its norm is 0

        normalized_semantics = normalize_vector(semantics)
        if np.isnan(normalized_semantics).any() or np.isinf(normalized_semantics).any():
            return
        semantics_hash = tuple(normalized_semantics)
        if (
            semantics_hash in self.seen_semantics
            and len(tree) > self.seen_semantics[semantics_hash]
        ):
            # self.seen_semantics_counter += 1
            # if self.seen_semantics_counter % 1000 == 0:
            #     print(self.seen_semantics_counter, len(self.trees))
            return

        self.seen_semantics[semantics_hash] = len(tree)
        self.trees.append(tree)
        self.normalized_semantics_list.append(
            normalized_semantics
        )  # Store the normalized semantics

    def clean_when_full(self, normalized_target_semantics):
        # Handle excess trees
        if len(self.trees) > self.max_trees:
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
            else:
                raise ValueError("Invalid updating mode")
            # Remove the oldest trees
            self.trees = [self.trees[idx] for idx in indexes]
            self.normalized_semantics_list = [
                self.normalized_semantics_list[idx] for idx in indexes
            ]
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

    def retrieve_nearest_tree(self, semantics: np.ndarray, return_semantics=False):
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
        if dist_neg < dist:
            dist = dist_neg
            index = index_neg

        self.plot_distance_function(dist)
        # if self.library_updating_mode == "LeastFrequentUsed":
        self.frequency[index] += 1
        if return_semantics:
            return self.trees[index], self.normalized_semantics_list[index]
        return self.trees[index]  # Return the corresponding tree

    def retrieve_smallest_nearest_tree(
        self,
        semantics: np.ndarray,
        return_semantics=False,
        top_k=10,
        incumbent_size=math.inf,
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
        dist, index = self.kd_tree.query(semantics, k=top_k)
        dist_neg, index_neg = self.kd_tree.query(-semantics, k=top_k)
        index = np.concatenate([index, index_neg])
        # From short to long
        sorted_index = np.argsort(np.concatenate([dist, dist_neg]))
        index = index[sorted_index][:top_k]

        smallest_index = -1
        for idx in range(top_k):
            if len(self.trees[index[idx]]) <= incumbent_size:
                smallest_index = index[idx]
                break

        if smallest_index == -1:
            smallest_index = np.argmin([len(self.trees[idx]) for idx in index])

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

    def update_forbidden_list(self, pop, total_features, feature_selection_mode):
        # self.group_selection(pop, total_features)
        # Count feature usage in each group
        features = defaultdict(float)
        for ind in pop:
            # Currently only supporting Linear Regression with LOOCV
            assert np.allclose(
                feature_importance_process(abs(ind.pipe["Ridge"].coef_)), ind.coef
            )
            for coef, tree in zip(ind.coef, ind.gene):
                for node in tree:
                    if isinstance(node, Terminal) and node.name.startswith("ARG"):
                        feature_id = int(node.name.replace("ARG", ""))
                        features[feature_id] += coef * ind.fitness.wvalues[0]

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
        for f_id in range(total_features):
            if f_id not in top_feature_ids:
                self.forbidden_list.append(f"ARG{f_id}")

        if self.verbose:
            # Print the top 95% features
            print(f"Top {int(feature_selection_mode * 100)}% features:")
            for feature_id, importance in top_features:
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
                self.forbidden_list.append(f"ARG{f_id}")

    def update_hard_instance(
        self,
        error: np.ndarray,
        semantics: np.ndarray,
        mode: str,
        features: np.ndarray,
        label: np.ndarray,
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
        elif mode == "Worst":
            errors = np.median(error, axis=0)
            self.clustering_indexes = np.argsort(errors)[-self.semantics_length :]
        elif mode == "Random":
            self.clustering_indexes = np.random.choice(
                np.arange(error.shape[1]), self.semantics_length, replace=False
            )
        else:
            raise Exception("Invalid mode")
        self.clear_all()

    def clear_all(self):
        self.frequency.clear()
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
