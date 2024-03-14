import math
from typing import List

import numpy as np
from scipy.spatial import cKDTree
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
        library_updating_mode="Recent",
        semantics_length=5,
        **params
    ):
        self.library_updating_mode = library_updating_mode
        self.kd_tree = None  # This will be a cKDTree instance
        self.trees = []  # List to store PrimitiveTree objects
        self.tree_map = {}  # Maps tree indices to PrimitiveTree objects
        self.normalized_semantics_list = []  # List to store normalized semantics
        self.seen_semantics = (
            set()
        )  # Set to store hashes of seen semantics for uniqueness
        self.max_trees = max_trees  # Maximum number of trees to store
        self.semantics_length = (
            semantics_length  # Length of semantics to use for KD-Tree
        )

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
                self.tree_map[len(self.trees) - 1] = tree
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
            else:
                raise ValueError("Invalid updating mode")
            # Remove the oldest trees
            self.trees = [self.trees[idx] for idx in indexes]
            self.normalized_semantics_list = [
                self.normalized_semantics_list[idx] for idx in indexes
            ]
            self.tree_map = {i: tree for i, tree in enumerate(self.trees)}
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
        return self.tree_map[index]  # Return the corresponding tree
