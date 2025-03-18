import random
from collections import defaultdict

import numpy as np


def get_random_downsampled_cases(population, downsample_rate):
    """
    Random Down-Sampling: Selects a random subset of training cases.
    - Simply selects `downsample_rate * num_cases` random cases.
    """
    num_cases = len(population[0].case_values)  # Number of training cases
    num_selected = int(downsample_rate * num_cases)  # Number of cases to select

    # Randomly select cases
    selected_cases = random.sample(range(num_cases), num_selected)

    return selected_cases  # Return indices of selected cases


def llm_selection(population, k=100):
    """
    Adaptive Multi-Dimensional Tournament Selection Operator.

    Args:
        population (list): List of individuals, each having a `case_values` attribute (numpy array).
        k (int): Number of individuals to select.

    Returns:
        list: Selected individuals.
    """
    if not population:
        return []

    population_size = len(population)
    num_cases = len(population[0].case_values)
    case_matrix = np.array(
        [ind.case_values for ind in population]
    )  # Shape: (population_size, num_cases)

    # Parameters
    tournament_size = 7
    num_subsets = 10  # Number of case subsets for multi-dimensional evaluation
    subset_size = max(1, num_cases // num_subsets)

    # Randomly shuffle case indices and partition into subsets
    shuffled_cases = np.random.permutation(num_cases)
    subsets = [
        shuffled_cases[i * subset_size : (i + 1) * subset_size]
        for i in range(num_subsets)
    ]

    # Compute aggregated fitness for each individual across all subsets
    # Here, we use the mean performance over all cases
    aggregated_fitness = np.mean(case_matrix, axis=1)  # Assuming lower is better

    # Initialize selection pool
    selected_indices = []
    niche_counts = defaultdict(int)

    for _ in range(k):
        # Perform a tournament
        competitors_idx = np.random.choice(
            population_size, size=tournament_size, replace=False
        )
        # Select the competitor with the best (lowest) aggregated fitness
        best_idx = competitors_idx[np.argmin(aggregated_fitness[competitors_idx])]

        # Determine the niche based on which subset this individual performs best
        individual_performance = case_matrix[best_idx]
        best_subset_idx = np.argmax(
            [np.mean(individual_performance[subset]) for subset in subsets]
        )
        niche_counts[best_subset_idx] += 1

        # Penalize selection probability for overrepresented niches
        niche_penalty = 1.0 / (1.0 + niche_counts[best_subset_idx])
        adjusted_fitness = aggregated_fitness[competitors_idx] * niche_penalty

        # Re-select the best after penalty
        best_idx = competitors_idx[np.argmin(adjusted_fitness)]
        selected_indices.append(best_idx)

    # Retrieve selected individuals
    selected_individuals = [population[idx] for idx in selected_indices]

    return selected_individuals


def llm_selection_plus(population, k=1):
    """
    Fitness-Diversity Weighted Tournament Selection Operator for Genetic Programming.

    This operator selects individuals based on a combination of their fitness and diversity,
    ensuring that selected individuals are both strong and diverse to promote effective crossover.

    Args:
        population (list): List of individuals, each having a `case_values` attribute (numpy array).
        k (int): Number of individuals to select.

    Returns:
        list: Selected individuals.
    """
    if not population or k <= 0:
        return []

    population_size = len(population)
    k = min(k, population_size)
    num_cases = len(population[0].case_values)

    # --- Fitness Calculation ---
    # Assume lower total errors are better
    epsilon = 1e-10  # Prevent division by zero
    total_errors = np.array(
        [np.sum(ind.case_values) for ind in population]
    )  # Shape: (population_size,)
    fitness = 1.0 / (total_errors + epsilon)  # Higher is better

    # Normalize fitness to [0, 1]
    fitness_min, fitness_max = fitness.min(), fitness.max()
    if fitness_max - fitness_min > epsilon:
        fitness_norm = (fitness - fitness_min) / (fitness_max - fitness_min)
    else:
        fitness_norm = np.ones_like(fitness)

    # --- Diversity Calculation ---
    # Select a subset of individuals as landmarks to approximate diversity
    num_landmarks = max(
        10, int(0.1 * population_size)
    )  # At least 10 landmarks or 10% of population
    landmark_indices = np.random.choice(
        population_size, size=num_landmarks, replace=False
    )
    landmarks = np.array(
        [population[idx].case_values for idx in landmark_indices]
    )  # Shape: (num_landmarks, num_cases)

    # Compute Euclidean distance of each individual to all landmarks and take the minimum distance
    # This approximates the uniqueness of each individual
    # Using broadcasting for efficient computation
    case_matrix = np.array(
        [ind.case_values for ind in population]
    )  # Shape: (population_size, num_cases)
    # Expand dimensions for broadcasting
    case_expanded = case_matrix[
        :, np.newaxis, :
    ]  # Shape: (population_size, 1, num_cases)
    landmarks_expanded = landmarks[
        np.newaxis, :, :
    ]  # Shape: (1, num_landmarks, num_cases)
    # Compute squared differences
    diff_squared = (
        case_expanded - landmarks_expanded
    ) ** 2  # Shape: (population_size, num_landmarks, num_cases)
    distances = np.sqrt(
        np.sum(diff_squared, axis=2)
    )  # Shape: (population_size, num_landmarks)
    min_distances = distances.min(axis=1)  # Shape: (population_size,)

    # Normalize diversity to [0, 1]
    diversity_min, diversity_max = min_distances.min(), min_distances.max()
    if diversity_max - diversity_min > epsilon:
        diversity_norm = (min_distances - diversity_min) / (
            diversity_max - diversity_min
        )
    else:
        diversity_norm = np.ones_like(min_distances)

    # --- Composite Scoring ---
    fitness_weight = 0.7
    diversity_weight = 0.3
    composite_score = fitness_weight * fitness_norm + diversity_weight * diversity_norm

    # Normalize composite scores to sum to 1 for probabilistic selection
    score_sum = composite_score.sum()
    if score_sum > epsilon:
        selection_probs = composite_score / score_sum
    else:
        selection_probs = np.full(population_size, 1.0 / population_size)

    # --- Tournament Selection with Diversity-Adaptive Probability ---
    tournament_size = 5  # Size of each tournament

    # Precompute cumulative probabilities for efficient sampling
    cumulative_probs = np.cumsum(selection_probs)

    selected = []
    for _ in range(k):
        # Randomly select tournament_size individuals based on selection_probs
        rand_values = np.random.rand(tournament_size)
        selected_indices = np.searchsorted(cumulative_probs, rand_values)
        selected_indices = np.clip(selected_indices, 0, population_size - 1)

        # Select the individual with the highest composite score in the tournament
        tournament_scores = composite_score[selected_indices]
        winner_idx = selected_indices[np.argmax(tournament_scores)]
        selected.append(population[winner_idx])

    return selected


def half_lexicase_selection_std(population, k=1):
    if not population:
        return []

    case_values = np.array([ind.case_values for ind in population])
    n_cases = case_values.shape[1]
    selected_individuals = []
    unique_selected = set()

    while len(selected_individuals) < k:
        # Shuffle case indices for diversity
        cases = list(range(n_cases))
        random.shuffle(cases)

        candidates_indices = list(range(len(population)))

        for case in cases:
            if len(candidates_indices) <= 1:
                break

            current_case_values = case_values[candidates_indices, case]
            min_case_value = np.min(current_case_values)

            # Dynamic threshold using the mean and standard deviation
            threshold = min_case_value + np.std(current_case_values) * 0.5

            # Filter candidates based on the computed threshold
            candidates_indices = [
                i for i in candidates_indices if case_values[i, case] <= threshold
            ]

            # Reset if no candidates remain to ensure robustness
            if not candidates_indices:
                candidates_indices = list(range(len(population)))
                break

        # Select a candidate and ensure uniqueness
        for idx in candidates_indices:
            if id(population[idx]) not in unique_selected:
                selected_individuals.append(population[idx])
                unique_selected.add(id(population[idx]))
                break

    return selected_individuals


def random_ds_tournament_selection(population, k, tournsize, downsample_rate=0.1):
    """
    Tournament Selection with Random Down-Sampling:
    - Selects a random subset of training cases.
    - Performs tournament selection using only the down-sampled cases.
    """
    selected = []

    for _ in range(k):  # Select k parents
        selected_cases = get_random_downsampled_cases(
            population, downsample_rate
        )  # Get random training case indices

        aspirants = random.sample(
            population, tournsize
        )  # Randomly pick `tournsize` individuals

        # Select winner based on down-sampled cases only
        winner = min(
            aspirants, key=lambda ind: np.mean(ind.case_values[selected_cases])
        )
        selected.append(winner)

    return selected  # Return selected individuals for reproduction
