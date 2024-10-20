from operator import eq

import numpy as np
from deap.tools import HallOfFame

from evolutionary_forest.model.clustering.shapley_pruning import (
    prune_models_based_on_shapley_for_regression,
)


class ShapleyPrunedHallOfFame(HallOfFame):
    def __init__(self, maxsize, y, similar=eq):
        """
        Inherit from HallOfFame and add the true labels y for Shapley pruning.
        :param maxsize: Maximum number of individuals to keep.
        :param y: True labels for the validation set (used in Shapley pruning).
        :param similar: Function to compare individuals, defaults to operator.eq.
        """
        super().__init__(maxsize, similar)
        self.y = y  # Store true labels for Shapley-based pruning

    def update(self, population):
        """
        Override the update method to include Shapley value-based pruning.
        """
        # First, update the hall of fame using the original logic
        super().update(population)

        # After updating the hall of fame, apply Shapley value pruning
        if len(self.items) > 0:
            # Extract predicted values from the individuals in the hall of fame
            new_hof = self.items  # Get the current hall of fame
            predictions = np.array(
                [ind.predicted_values for ind in new_hof]
            )  # Extract predictions

            # Prune based on Shapley values
            index = prune_models_based_on_shapley_for_regression(predictions, self.y)

            # Update the hall of fame with only the pruned individuals
            self.items = [new_hof[i] for i in index]
            self.keys = [
                new_hof[i].fitness for i in index
            ]  # Update the fitness keys as well
