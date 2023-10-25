import unittest

from evolutionary_forest.component.bloat_control.depth_limit import *


class TestFunctions(unittest.TestCase):
    def test_get_replacement_tree(self):
        class Individual:
            def __init__(self, gene):
                self.gene = gene

        # Test case 1: When there are unused trees, a random choice should be made.
        parent = [1, 2, 3, 4, 5]
        ind = Individual([1, 2, 3])
        random.seed(0)  # Seed for reproducibility
        replacement = get_replacement_tree(ind, parent)
        self.assertIn(replacement, parent)
        self.assertNotIn(replacement, ind.gene)

        # Test case 2: When all trees are used, it should return None.
        parent = [1, 2, 3]
        ind = Individual([1, 2, 3])
        random.seed(0)  # Reset seed for reproducibility
        replacement = get_replacement_tree(ind, parent)
        self.assertIsNone(replacement)

    def test_remove_none_values(self):
        # Test case 1: Basic case with None values in the list.
        my_list = [1, None, 3, 4, None, 6]
        filtered_list = remove_none_values(my_list)
        self.assertEqual(filtered_list, [1, 3, 4, 6])

        # Test case 2: List with no None values should remain unchanged.
        my_list = [1, 2, 3, 4, 5]
        filtered_list = remove_none_values(my_list)
        self.assertEqual(filtered_list, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
