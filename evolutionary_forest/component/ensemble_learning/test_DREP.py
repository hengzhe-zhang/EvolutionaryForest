import unittest

import numpy as np

from evolutionary_forest.component.ensemble_learning.DREP import DREPEnsemble


class TestDREPEnsemble(unittest.TestCase):

    def setUp(self):
        # Mocked data for the purpose of the test
        self.y_data = np.array([1, 2, 3, 4, 5])
        self.predictions = [
            np.array([0.8, 2.2, 2.7, 3.9, 5.1]),
            np.array([1.1, 1.9, 3.1, 3.8, 5.2]),
            np.array([1.3, 2.0, 3.0, 4.2, 4.8])
        ]

        self.algorithm = lambda: None  # Simple mock of the algorithm object
        self.algorithm.hof = [1, 2, 3]  # Mocking Hall of Fame attribute

    def test_generate_ensemble_weights(self):
        ensemble = DREPEnsemble(self.y_data, self.predictions, self.algorithm)
        ensemble.generate_ensemble_weights()
        weights = ensemble.get_ensemble_weights()

        # Check if the sum of weights is close enough to 1 (due to possible float inaccuracies)
        self.assertTrue(np.isclose(np.sum(weights), 1.0, atol=1e-8))

        # Check if weights are between 0 and 1
        for weight in weights:
            self.assertTrue(0 <= weight <= 1)


if __name__ == "__main__":
    unittest.main()
