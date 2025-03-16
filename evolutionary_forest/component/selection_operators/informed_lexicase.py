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


def adaptive_diverse_selection(population, k=1):
    # Adaptive threshold-based clustering for population diversity
    diversity_threshold = 0.25

    def compute_score(ind):
        """Compute combined score based on normalized error and diversity aspect."""
        errors = ind.case_values
        diversity_score = np.std(
            errors
        )  # Diversity via the standard deviation across cases
        return np.mean(errors) + diversity_threshold * diversity_score

    # Compute scores for all individuals
    scores = [(compute_score(ind), idx) for idx, ind in enumerate(population)]
    scores.sort()  # Sort by combined scores (lower is better)

    selected_indices = set()
    selected_individuals = []

    while len(selected_individuals) < k:
        # Pick the best-ranked individual for high exploitation value
        best_score, best_idx = scores.pop(0)

        if best_idx in selected_indices:
            continue  # Avoid duplicates

        selected_individuals.append(population[best_idx])
        selected_indices.add(best_idx)

        # Adjust scores dynamically to foster diversity in the next selection
        for i, (score, idx) in enumerate(scores):
            if idx not in selected_indices:
                # Increase the effective score of similar individuals to improve diversity
                scores[i] = (score + diversity_threshold * np.random.random(), idx)

        scores.sort()  # Re-sort based on updated scores

    return selected_individuals[:k]


def diverse_performance_selection(population, k=1):
    def calculate_diversity(individual):
        """Calculate diversity based on the variance of performance."""
        return 1.0 / (1.0 + np.var(individual.case_values))

    def calculate_similarity(ind1, ind2):
        """Calculate structural and performance similarity."""
        structural_similarity = 1.0 / (
            1.0 + abs(len(ind1.case_values) - len(ind2.case_values))
        )
        performance_similarity = np.mean(
            np.abs(np.array(ind1.case_values) - np.array(ind2.case_values))
        )
        return structural_similarity * (1.0 / (1.0 + performance_similarity))

    num_cases = len(population[0].case_values)
    selected_individuals = []

    while len(selected_individuals) < k:
        candidates = population[:]
        cases = np.random.permutation(num_cases)

        for case_index in cases:
            errors = np.array([ind.case_values[case_index] for ind in candidates])
            min_error = np.min(errors)
            candidates = [
                ind for ind, error in zip(candidates, errors) if error == min_error
            ]
            if len(candidates) == 1:
                break

        if len(candidates) > 1:
            diversity_scores = [calculate_diversity(ind) for ind in candidates]
            best_diversity_idx = np.argmax(diversity_scores)
            baseline = candidates[best_diversity_idx]

            sorted_candidates = sorted(
                candidates,
                key=lambda ind: calculate_similarity(baseline, ind),
                reverse=True,
            )

            selected_pair = [
                baseline,
                sorted_candidates[0],
            ]  # Choose the most similar to the baseline
            if len(selected_pair) < 2:
                selected_pair.append(random.choice(candidates))

            selected_individuals.extend(selected_pair[:2])
        else:
            selected_individuals.append(candidates[0])

    return selected_individuals[:k]


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
