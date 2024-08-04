import unittest
import numpy as np
from deap.gp import PrimitiveTree

from evolutionary_forest.utility.tree_pool import *


class TestSemanticLibrary(unittest.TestCase):
    def setUp(self):
        self.library = SemanticLibrary(max_trees=5, semantics_length=3)
        self.semantics = np.array([1.0, 2.0, 3.0])
        self.tree = PrimitiveTree("ADD")

    def test_append_semantics(self):
        self.library.append_subtree(self.semantics, self.tree)
        self.assertEqual(len(self.library.trees), 1)
        self.assertEqual(len(self.library.normalized_semantics_list), 1)
        self.assertIn(
            tuple(self.semantics / np.linalg.norm(self.semantics)),
            self.library.seen_semantics,
        )

    def test_clean_when_full(self):
        # Add trees to fill the library
        for i in range(6):
            semantics = np.array([i, i + 1, i + 2])
            tree = PrimitiveTree(f"ADD_{i}")
            self.library.append_subtree(semantics, tree)
        self.library.clean_when_full(self.semantics)
        self.assertEqual(len(self.library.trees), 5)
        self.assertEqual(len(self.library.normalized_semantics_list), 5)

    def test_retrieve_nearest_tree(self):
        # Add trees to fill the library
        for i in range(3):
            semantics = np.array([i, i + 1, i + 2])
            tree = PrimitiveTree(f"ADD_{i}")
            self.library.append_subtree(semantics, tree)
        self.library.append_full_tree([], self.semantics)
        nearest_tree = self.library.retrieve_nearest_tree(self.semantics)
        self.assertIsNotNone(nearest_tree)

    def test_update_by_map_elites(self):
        semantics_list = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0]),
        ]
        target_semantics = np.array([3.0, 3.0, 3.0])
        indices = self.library.update_by_map_elites(semantics_list, target_semantics)
        self.assertIsNotNone(indices)

    def test_retrieve_smallest_nearest_tree(self):
        # Add trees to fill the library
        for i in range(3):
            semantics = np.array([i, i + 1, i + 2])
            tree = PrimitiveTree(f"ADD_{i}")
            self.library.append_subtree(semantics, tree)
        self.library.append_full_tree([], self.semantics)
        nearest_tree = self.library.retrieve_smallest_nearest_tree(self.semantics)
        self.assertIsNotNone(nearest_tree)

    def test_select_samples_via_quantiles(self):
        y = np.random.randn(1000)
        selected_indices = select_samples_via_quantiles(y)
        self.assertEqual(len(selected_indices), 100)
        self.assertTrue(np.issubdtype(type(selected_indices[0]), np.integer))


if __name__ == "__main__":
    unittest.main()
