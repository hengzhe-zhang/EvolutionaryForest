import random
from collections import defaultdict

import numpy as np

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast
from evolutionary_forest.model.optimal_knn.GBOptimalKNN import RidgeBoostedKNN
from evolutionary_forest.multigene_gp import MultipleGeneGP


def selCompetitive(pop, k):
    # Initialize count and inds dictionaries
    count = defaultdict(int)
    inds = defaultdict(list)

    # Group individuals by n_neighbors
    for ind in pop:
        ind: MultipleGeneGP
        assert isinstance(ind.pipe["Ridge"], RidgeBoostedKNN)
        n_neighbors = ind.pipe["Ridge"].knn_params.get("n_neighbors", 5)
        count[n_neighbors] += 1
        inds[n_neighbors].append(ind)

    # Get all unique neighbor values
    neighbor_values = sorted(count.keys())

    # Calculate normalized counts (probabilities)
    total_count = sum(count.values())
    normalized_counts = {n: count[n] / total_count for n in neighbor_values}

    selection = []

    # Select k//2 groups and apply epsilon lexicase selection
    num_groups = k // 2

    for _ in range(num_groups):
        # Select group based on normalized count (probability proportional selection)
        weights = [normalized_counts[n] for n in neighbor_values]
        selected_group = random.choices(neighbor_values, weights=weights, k=1)[0]

        # Apply epsilon lexicase selection to get first parent
        parent_a = selAutomaticEpsilonLexicaseFast(inds[selected_group], 1)[0]

        # Find the best candidate that complements parent_a
        best_candidate = None
        best_value = float("inf")  # Initialize with infinity (looking for minimum)

        for candidate in pop:
            # Calculate complementary score (element-wise minimum sum)
            current_value = np.sum(
                np.minimum(parent_a.case_values, candidate.case_values)
            )

            if current_value < best_value:
                best_value = current_value
                best_candidate = candidate

        selection.append(parent_a)
        selection.append(best_candidate)

    return selection
