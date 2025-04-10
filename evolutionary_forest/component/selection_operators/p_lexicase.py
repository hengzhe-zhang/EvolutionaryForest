import numpy as np
from scipy.special import softmax


def n_plexicase_selection(fitness_matrix, temperature=1.0, num_parents=1):
    """
    n-Plexicase selection using MAD-based relaxation (for minimization tasks).

    Args:
        fitness_matrix: numpy array of shape (num_individuals, num_cases)
                        where lower is better (minimization).
        temperature: float, controls sharpness of selection distribution.
        num_parents: int, number of individuals to select.

    Returns:
        List of selected individual indices.
    """
    num_individuals, num_cases = fitness_matrix.shape

    # Step 1: Compute per-case MAD thresholds
    medians = np.median(fitness_matrix, axis=0)
    mad = np.median(np.abs(fitness_matrix - medians), axis=0)

    # Step 2: Determine elites using MAD-based relaxed threshold
    best_per_case = np.min(fitness_matrix, axis=0)
    relaxed_threshold = best_per_case + mad
    elite_mask = fitness_matrix <= relaxed_threshold  # shape: (indivs, cases)

    # Step 3: Count how often each individual is elite
    elitism_counts = elite_mask.sum(axis=1)  # shape: (num_individuals,)
    eligible = elitism_counts > 0

    if not np.any(eligible):
        raise ValueError("No eligible individuals found under MAD threshold.")

    # Step 4: Compute selection probabilities using temperature scaling
    raw_scores = np.where(eligible, elitism_counts, -np.inf)
    probs = softmax(raw_scores * temperature)

    # Step 5: Sample parents based on probabilities
    selected = np.random.choice(
        num_individuals, size=num_parents, p=probs, replace=False
    )
    return selected.tolist()


if __name__ == "__main__":
    # Example fitness values for 5 individuals over 4 cases
    fitness = np.array(
        [
            [0.8, 0.9, 0.85, 0.7],
            [0.82, 0.85, 0.88, 0.69],
            [0.75, 0.9, 0.84, 0.68],
            [0.8, 0.8, 0.87, 0.72],
            [0.7, 0.85, 0.83, 0.7],
        ]
    )

    selected = n_plexicase_selection(
        fitness_matrix=fitness, temperature=1.0, num_parents=2
    )
    print("Selected individuals:", selected)
