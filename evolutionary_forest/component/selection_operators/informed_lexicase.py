import random

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
    Adaptive Fitness-Diversity Balanced Selection (AFDBS)

    Selects a balanced set of high-fitness and diverse individuals to promote
    effective crossover operations. Ensures selected parents are both strong
    performers and genetically diverse.

    Args:
        population (list): List of individuals, each having a 'case_values' attribute (numpy array).
        k (int): Number of individuals to select.

    Returns:
        selected_individuals (list): List of selected individuals.
    """
    if not population or k <= 0:
        return []

    num_individuals = len(population)
    case_matrix = np.array([ind.case_values for ind in population])  # Shape: (n, cases)

    # Compute Fitness: Inverse of mean error
    epsilon = 1e-10
    mean_errors = case_matrix.mean(axis=1)
    fitness = 1.0 / (mean_errors + epsilon)

    # Normalize Fitness to [0, 1]
    fitness_norm = (fitness - fitness.min()) / (fitness.max() - fitness.min() + epsilon)

    # Compute Diversity: Distance from population centroid
    centroid = case_matrix.mean(axis=0)
    diversity = np.linalg.norm(case_matrix - centroid, axis=1)

    # Normalize Diversity to [0, 1]
    diversity_norm = (diversity - diversity.min()) / (
        diversity.max() - diversity.min() + epsilon
    )

    # Adaptive Weighting based on diversity variance
    diversity_variance = np.var(diversity_norm)
    if diversity_variance < 0.01:
        alpha = 0.5  # Increase diversity weight
    else:
        alpha = 0.7  # More emphasis on fitness
    beta = 1.0 - alpha

    # Composite Score
    composite_scores = alpha * fitness_norm + beta * diversity_norm

    # Normalize Composite Scores to form initial selection probabilities
    total_score = composite_scores.sum()
    if total_score == 0:
        selection_prob = np.ones(num_individuals) / num_individuals
    else:
        selection_prob = composite_scores / total_score

    selected_indices = []
    selected_vectors = []

    # Precompute normalized case_values for similarity computation
    norms = np.linalg.norm(case_matrix, axis=1, keepdims=True) + epsilon
    normalized_cases = case_matrix / norms  # Shape: (n, cases)

    for _ in range(k):
        if selection_prob.sum() == 0:
            # If all probabilities are zero, select uniformly at random
            chosen_idx = np.random.choice(num_individuals)
        else:
            chosen_idx = np.random.choice(num_individuals, p=selection_prob)

        selected_indices.append(chosen_idx)
        selected_vectors.append(normalized_cases[chosen_idx])

        # Update selection probabilities to penalize similar individuals
        if len(selected_vectors) == 1:
            # First selection, no penalty yet
            last_selected = normalized_cases[chosen_idx]
        else:
            # Calculate similarity with the last selected individual
            similarity = np.dot(normalized_cases, selected_vectors[-1])
            diversity_strength = 0.3  # Controls the impact of diversity (0 to 1)
            selection_prob *= 1 - diversity_strength * similarity
            selection_prob = np.clip(selection_prob, a_min=0, a_max=None)
            # Re-normalize probabilities
            if selection_prob.sum() > 0:
                selection_prob /= selection_prob.sum()
            else:
                # Reset to uniform if all probabilities are zero
                selection_prob = np.ones(num_individuals) / num_individuals

    # Retrieve the selected individuals
    selected_individuals = [population[idx] for idx in selected_indices]
    return selected_individuals


def llm_selection_plus_plus(population, k=1):
    """
    Fitness-Informed Diverse Selection (FIDS)

    Selects a diverse set of high-fitness individuals ensuring that selected parents are
    both strong performers and genetically diverse to promote effective crossover operations.

    Args:
        population (list): List of individuals, each having a 'case_values' attribute (numpy array).
        k (int): Number of individuals to select.

    Returns:
        selected_individuals (list): List of selected individuals.
    """
    if not population or k <= 0:
        return []

    population_size = len(population)
    num_cases = population[0].case_values.shape[0]

    # Compute fitness: lower mean of case_values is better
    fitness = np.array([np.mean(ind.case_values) for ind in population])
    inverted_fitness = 1.0 / (fitness + 1e-8)  # Avoid division by zero
    selection_prob = inverted_fitness / inverted_fitness.sum()

    # Precompute normalized case_values for cosine similarity
    case_matrix = np.array([ind.case_values for ind in population])  # Shape: (n, cases)
    norms = (
        np.linalg.norm(case_matrix, axis=1, keepdims=True) + 1e-8
    )  # Avoid division by zero
    normalized_cases = case_matrix / norms  # Shape: (n, cases)

    selected_indices = []
    for _ in range(k):
        if selection_prob.sum() == 0:
            # If all probabilities are zero, select randomly
            chosen_idx = np.random.choice(population_size)
        else:
            chosen_idx = np.random.choice(population_size, p=selection_prob)
        selected_indices.append(chosen_idx)

        # Update selection probabilities to reduce similarity with the selected individual
        selected_vector = normalized_cases[chosen_idx]
        similarity = np.dot(normalized_cases, selected_vector)
        # Dampen probabilities of similar individuals
        diversity_strength = 0.5  # Controls the impact of diversity (0 to 1)
        selection_prob *= 1 - diversity_strength * similarity
        # Ensure no negative probabilities
        selection_prob = np.clip(selection_prob, a_min=0, a_max=None)
        # Re-normalize probabilities
        if selection_prob.sum() > 0:
            selection_prob /= selection_prob.sum()
        else:
            # If all probabilities are zero, reset to uniform
            selection_prob = np.ones(population_size) / population_size

    # Retrieve the selected individuals
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
