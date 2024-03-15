import math
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree, KDTree
from scipy.special import softmax
from scipy.stats import pearsonr
from sklearn.decomposition import KernelPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evolutionary_forest.component.configuration import MAPElitesConfiguration
from evolutionary_forest.multigene_gp import MultipleGeneGP, map_elites_selection
from evolutionary_forest.utility.normalization_tool import normalize_vector


class TreePool:
    def __init__(
        self,
        max_trees=1000,
        library_updating_mode="LeastFrequentUsed",
        semantics_length=5,
        **params
    ):
        self.frequency = defaultdict(int)
        self.plot_mismatch = False
        self.plot_distance = False
        self.library_updating_mode = library_updating_mode
        self.kd_tree: KDTree = None  # This will be a cKDTree instance
        self.trees = []  # List to store PrimitiveTree objects
        self.normalized_semantics_list = []  # List to store normalized semantics
        # Set to store hashes of seen semantics for uniqueness
        self.seen_semantics = set()
        self.max_trees = max_trees  # Maximum number of trees to store
        # Length of semantics to use for KD-Tree
        self.semantics_length = semantics_length

        self.log_initialization()

    def log_initialization(self):
        self.mismatch_times = []
        self.distance_distribution = []

    def update_kd_tree(self, inds: List[MultipleGeneGP], target_semantics: np.ndarray):
        target_semantics = target_semantics[: self.semantics_length]
        normalized_target_semantics = normalize_vector(target_semantics)

        points = (
            []
        )  # This will store all unique and normalized semantics for the KD-Tree
        for ind in inds:
            for semantics, tree in zip(ind.semantics.T, ind.gene):
                semantics = semantics[
                    : self.semantics_length
                ]  # Assuming you want to use the first semantics_length elements
                # Normalize semantics and skip if norm is 0
                norm = np.linalg.norm(semantics)
                if norm == 0:
                    continue  # Skip this semantics as its norm is 0

                normalized_semantics = normalize_vector(semantics)
                semantics_hash = tuple(normalized_semantics)
                if semantics_hash in self.seen_semantics:
                    continue

                self.seen_semantics.add(semantics_hash)
                self.trees.append(tree)
                self.normalized_semantics_list.append(
                    normalized_semantics
                )  # Store the normalized semantics
                points.append(normalized_semantics)

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
            else:
                raise ValueError("Invalid updating mode")
            # Remove the oldest trees
            self.trees = [self.trees[idx] for idx in indexes]
            self.normalized_semantics_list = [
                self.normalized_semantics_list[idx] for idx in indexes
            ]
            # Recreate seen_semantics based on the current normalized_semantics_list
            self.seen_semantics = {
                tuple(semantics) for semantics in self.normalized_semantics_list
            }

        # Create the KDTree with all collected points
        if points:
            self.kd_tree = cKDTree(self.normalized_semantics_list)

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

    def retrieve_nearest_tree(self, semantics: np.ndarray):
        if self.kd_tree is None:
            raise ValueError("KD-Tree is empty. Please add some trees first.")

        semantics = semantics[: self.semantics_length]

        # Normalize the query semantics
        norm = np.linalg.norm(semantics)
        if norm > 0:
            semantics = semantics / norm
        else:
            raise ValueError("Query semantics norm is 0, cannot normalize.")

        # Query the KDTree for the nearest point
        dist, index = self.kd_tree.query(semantics)
        if self.library_updating_mode == "LeastFrequentUsed":
            self.frequency[index] += 1
        return self.trees[index]  # Return the corresponding tree

    def retrieve_nearest_trees_weighted(self, semantics: np.ndarray, top_k=10, std=1):
        if self.kd_tree is None:
            raise ValueError("KD-Tree is empty. Please add some trees first.")

        semantics = semantics[: self.semantics_length]

        # Normalize the query semantics
        norm = np.linalg.norm(semantics)
        if norm > 0:
            semantics = semantics / norm
        else:
            raise ValueError("Query semantics norm is 0, cannot normalize.")

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
        if self.plot_distance:
            self.distance_distribution.append(best_distance)
            if (
                len(self.distance_distribution) > 0
                and len(self.distance_distribution) % 100 == 0
            ):
                plt.hist(self.distance_distribution, bins=10, color="blue", alpha=0.7)
                plt.title("Distance Distribution")
                plt.xlabel("Distance")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
        if self.library_updating_mode == "LeastFrequentUsed":
            self.frequency[best_tree_index] += 1
        return self.trees[best_tree_index]  # Return the corresponding tree
