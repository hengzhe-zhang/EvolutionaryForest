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


def llm_selection(population, k=100, tour_size=7):
    import numpy as np
    import random

    def get_information(individual):
        mse_vector = np.array(individual.case_values)
        predicted_values = np.array(individual.predicted_values)
        residual = individual.y - predicted_values
        number_of_nodes = len(predicted_values)
        return mse_vector, predicted_values, residual, number_of_nodes

    # Calculate compatibility score between two individuals
    def compatibility_score(ind1, ind2):
        mse1, _, residual1, _ = get_information(ind1)
        mse2, _, residual2, _ = get_information(ind2)
        return np.abs(mse1.mean() - mse2.mean()) + np.dot(residual1, residual2)

    selected_individuals = []

    for _ in range(k // 2):
        # Select a subset of individuals for parent A
        tournament_a = random.sample(population, tour_size)
        # Select parent A based on lowest mean squared error (favoring specialization)
        parent_a = min(tournament_a, key=lambda ind: np.mean(get_information(ind)[0]))
        selected_individuals.append(parent_a)

        # For parent B, consider compatibility with parent A
        tournament_b = random.sample(population, tour_size)
        # Calculate compatibility for each tournament B candidate with selected parent A
        compatibility_scores = [
            compatibility_score(parent_a, ind) for ind in tournament_b
        ]
        # Select parent B as the most compatible with parent A
        parent_b = min(tournament_b, key=lambda ind: compatibility_score(parent_a, ind))
        selected_individuals.append(parent_b)

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
