import unittest
from unittest.mock import patch

import evolutionary_forest.component.selection as evo_selection


class MockIndividual:
    def __init__(self, fitness_value):
        self.fitness = MockFitness(fitness_value)


class MockFitness:
    def __init__(self, fitness_value):
        self.wvalues = (0, fitness_value)


class TestNicheBaseSelection(unittest.TestCase):
    @patch.object(evo_selection, "selAutomaticEpsilonLexicaseFast")
    def test_niche_base_selection(self, mock_sel):
        from evolutionary_forest.component.selection_operators.niche_base_selection import (
            niche_base_selection,
        )

        # Mock the return value of selAutomaticEpsilonLexicaseFast
        mock_sel.side_effect = lambda x, k: x[:k]

        # Create mock individuals with given fitness values
        mock_individuals = [MockIndividual(i) for i in range(10)]
        selected_individuals = niche_base_selection(mock_individuals, 3)

        # Selecting 3 individuals,
        self.assertEqual(len(selected_individuals), 3)
        print(selected_individuals)


if __name__ == "__main__":
    unittest.main()
