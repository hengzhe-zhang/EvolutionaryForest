import numpy as np
from scipy.special import softmax


def n_plexicase_selection(population, k, temperature=1.0):
    """
    n-Plexicase selection for DEAP-style individuals.

    Each individual must have a .case_values attribute (list/array of fitness or error values per case).

    Args:
        population: list of individuals with .case_values attribute (shape: num_cases).
        temperature: float, softmax temperature to control selection pressure.
        k: int, number of individuals to select.

    Returns:
        List of selected individuals.
    """
    num_individuals = len(population)
    if num_individuals == 0:
        return []

    # Step 1: Build fitness matrix from .case_values
    fitness_matrix = np.array([ind.case_values for ind in population])  # shape: (N, C)
    num_cases = fitness_matrix.shape[1]

    # Step 2: Compute MAD-based relaxed threshold (for minimization)
    medians = np.median(fitness_matrix, axis=0)
    mad = np.median(np.abs(fitness_matrix - medians), axis=0)
    best_per_case = np.min(fitness_matrix, axis=0)
    relaxed_threshold = best_per_case + mad

    # Step 3: Determine elite mask
    elite_mask = fitness_matrix <= relaxed_threshold  # (N, C)

    # Step 4: Count elite occurrences per individual
    elitism_counts = elite_mask.sum(axis=1)
    eligible = elitism_counts > 0

    if not np.any(eligible):
        raise ValueError("No eligible individuals found under MAD threshold.")

    # Step 5: Apply temperature scaling and softmax
    raw_scores = np.where(eligible, elitism_counts, -np.inf)
    probs = softmax(raw_scores * temperature)

    # Step 6: Sample parents based on the probability distribution
    selected_indices = np.random.choice(num_individuals, size=k, p=probs, replace=False)
    selected_individuals = [population[i] for i in selected_indices]
    return selected_individuals
