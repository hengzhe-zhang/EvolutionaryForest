from typing import List

import numpy as np
from scipy.spatial import cKDTree

from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utility.normalization_tool import normalize_vector


class TreePool:
    def __init__(self, max_trees=1000):
        self.kd_tree = None  # This will be a cKDTree instance
        self.trees = []  # List to store PrimitiveTree objects
        self.tree_map = {}  # Maps tree indices to PrimitiveTree objects
        self.normalized_semantics_list = []  # List to store normalized semantics
        self.seen_semantics = (
            set()
        )  # Set to store hashes of seen semantics for uniqueness
        self.max_trees = max_trees  # Maximum number of trees to store
        self.semantics_length = 5  # Length of semantics to use for KD-Tree

    def update_kd_tree(self, inds: List[MultipleGeneGP]):
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
            excess = len(self.trees) - self.max_trees
            # Remove the oldest trees
            self.trees = self.trees[excess:]
            self.normalized_semantics_list = self.normalized_semantics_list[excess:]
            self.tree_map = {i: tree for i, tree in enumerate(self.trees)}
            # Recreate seen_semantics based on the current normalized_semantics_list
            self.seen_semantics = {
                tuple(semantics) for semantics in self.normalized_semantics_list
            }

        # Create the KDTree with all collected points
        if points:
            self.kd_tree = cKDTree(self.normalized_semantics_list)

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
