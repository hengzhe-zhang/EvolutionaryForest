import random

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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
    Adaptive Performance-Diversity Selection (APDS) Operator for Genetic Programming.

    This operator adaptively balances individual performance with diversity to select
    a set of high-quality and varied solutions. It ensures that selected parents are
    both strong performers and diverse to promote effective crossover and prevent
    premature convergence.

    Parameters:
    - population: List of individuals. Each individual must have a 'case_values' attribute,
                 which is a numpy array of error values per test case.
    - k: Number of individuals to select.

    Returns:
    - selected_individuals: List of selected individuals.
    """
    if not population:
        return []

    population_size = len(population)
    num_cases = len(population[0].case_values)

    # Extract case_values into a 2D NumPy array
    case_matrix = np.array(
        [ind.case_values for ind in population]
    )  # Shape: (population_size, num_cases)

    # Step 1: Compute Performance Scores
    # Assuming minimization: lower total error is better
    total_errors = np.sum(case_matrix, axis=1)  # Shape: (population_size,)
    # Avoid division by zero
    total_errors = np.where(total_errors == 0, 1e-6, total_errors)
    performance_scores = 1.0 / total_errors  # Higher is better

    # Step 2: Compute Diversity Scores
    # Diversity based on the number of unique cases where the individual is a top performer
    top_p = 0.05  # Top 5% are considered top performers
    top_n = max(1, int(np.ceil(top_p * population_size)))

    # Argsort ascending (lower error is better)
    sorted_indices = np.argsort(
        case_matrix, axis=0
    )  # Shape: (population_size, num_cases)
    top_indices = sorted_indices[:top_n, :]  # Shape: (top_n, num_cases)

    # Create a binary matrix where 1 indicates the individual is a top performer for the case
    top_performance = np.zeros_like(case_matrix, dtype=float)
    row_indices = np.repeat(np.arange(top_n), num_cases)
    col_indices = np.tile(np.arange(num_cases), top_n)
    # Ensure indices are within bounds
    top_indices_flat = top_indices.flatten()
    col_indices_flat = np.tile(np.arange(num_cases), top_n)
    top_performance[top_indices_flat, col_indices_flat] = 1.0

    # Diversity score: number of cases where individual is a top performer
    diversity_scores = np.sum(top_performance, axis=1)  # Shape: (population_size,)

    # Step 3: Normalize Performance and Diversity Scores
    # Normalize performance
    perf_min = performance_scores.min()
    perf_max = performance_scores.max()
    if perf_max > perf_min:
        normalized_performance = (performance_scores - perf_min) / (perf_max - perf_min)
    else:
        normalized_performance = np.ones_like(performance_scores)

    # Normalize diversity
    div_min = diversity_scores.min()
    div_max = diversity_scores.max()
    if div_max > div_min:
        normalized_diversity = (diversity_scores - div_min) / (div_max - div_min)
    else:
        normalized_diversity = np.ones_like(diversity_scores)

    # Step 4: Adaptive Weighting Based on Population Diversity
    # Compute overall population diversity
    population_diversity = np.mean(diversity_scores)
    # Define a threshold to decide when to prioritize diversity
    diversity_threshold = (div_max - div_min) * 0.5 + div_min
    if population_diversity < diversity_threshold:
        weight_diversity = 0.6
        weight_performance = 0.4
    else:
        weight_diversity = 0.3
        weight_performance = 0.7

    # Step 5: Combine Normalized Scores
    combined_scores = (weight_performance * normalized_performance) + (
        weight_diversity * normalized_diversity
    )

    # Step 6: Compute Selection Probabilities
    total_combined = np.sum(combined_scores)
    if total_combined == 0 or not np.isfinite(combined_scores).all():
        selection_probs = np.ones(population_size) / population_size
    else:
        selection_probs = combined_scores / total_combined

    # Step 7: Select k Individuals Based on Probabilities
    selected_indices = np.random.choice(
        population_size, size=k, replace=True, p=selection_probs
    )
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


def llm_selection_plus_plus(population, k=1):
    """
    Adaptive Diversity-Driven Selection (ADDS) Operator for Genetic Programming.

    This operator selects a diverse and high-performing set of individuals while ensuring
    that selected parents are compatible for effective crossover. It dynamically balances
    selection pressure based on population diversity to promote both exploration and exploitation.

    Parameters:
    - population: List of individuals, each having a `case_values` attribute
                  which is a NumPy array of error values across test cases.
    - k: Number of individuals to select.

    Returns:
    - selected_individuals: List of selected individuals.
    """
    if not population or k <= 0:
        return []

    # Extract case_values into a 2D NumPy array
    case_matrix = np.array(
        [ind.case_values for ind in population]
    )  # Shape: (n_individuals, n_cases)

    # Compute fitness as inverse of mean error
    mean_errors = case_matrix.mean(axis=1)
    epsilon = 1e-10  # Prevent division by zero
    fitness_scores = 1.0 / (mean_errors + epsilon)

    # Normalize fitness scores to [0, 1]
    fitness_min, fitness_max = fitness_scores.min(), fitness_scores.max()
    if fitness_max > fitness_min:
        norm_fitness = (fitness_scores - fitness_min) / (fitness_max - fitness_min)
    else:
        norm_fitness = np.ones_like(fitness_scores)

    # Compute diversity based on cosine similarity of case performance
    similarity_matrix = cosine_similarity(case_matrix)
    # Diversity score inversely related to average similarity with others
    diversity_scores = 1 - similarity_matrix.mean(axis=1)  # Shape: (n_individuals,)

    # Normalize diversity scores to [0, 1]
    diversity_min, diversity_max = diversity_scores.min(), diversity_scores.max()
    if diversity_max > diversity_min:
        norm_diversity = (diversity_scores - diversity_min) / (
            diversity_max - diversity_min
        )
    else:
        norm_diversity = np.ones_like(diversity_scores)

    # Adaptive weighting based on population diversity
    population_diversity = diversity_scores.mean()
    if population_diversity > 0.5:
        # Prefer diversity more
        alpha = 0.4
    else:
        # Favor fitness more
        alpha = 0.7

    combined_scores = alpha * norm_fitness + (1 - alpha) * norm_diversity
    combined_scores = np.clip(combined_scores, a_min=0, a_max=None)

    # If all scores are zero, fallback to uniform probabilities
    total = combined_scores.sum()
    if total == 0:
        selection_probs = np.ones(len(population)) / len(population)
    else:
        selection_probs = combined_scores / total

    # Select individuals based on selection probabilities
    selected_indices = np.random.choice(
        len(population), size=k, replace=True, p=selection_probs
    )
    selected_individuals = [population[idx] for idx in selected_indices]

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
