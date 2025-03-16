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
    def lexicase_filter(individuals, num_cases):
        """Adaptive filtering based on lexicase principles."""
        case_indices = np.random.permutation(num_cases)
        candidates = individuals[:]
        for idx in case_indices:
            min_error = min(ind.case_values[idx] for ind in candidates)
            candidates = [
                ind for ind in candidates if ind.case_values[idx] == min_error
            ]
            if len(candidates) <= 1:
                break
        return candidates

    def synergy_score(reference, target):
        """Evaluates synergy considering complementarity."""
        return np.mean(1.0 / (1.0 + np.abs(reference - target)))

    def entropy_score(ind):
        """Calculates entropy for assessing diversity."""
        values = np.array(ind.case_values)
        return -np.sum(values * np.log(values + 1e-9))

    num_cases = len(population[0].case_values)
    selected = []

    while len(selected) < k:
        # Step 1: Adaptive filtering
        candidates = lexicase_filter(population, num_cases)

        if len(candidates) == 1:
            selected.append(candidates[0])
            continue

        # Step 2: Calculate diversity and synergy
        reference = candidates[0]
        entropy_scores = [entropy_score(ind) for ind in candidates]
        synergy_scores = [
            synergy_score(reference.case_values, ind.case_values) for ind in candidates
        ]

        # Combine diversity with synergy to determine selection
        combined_scores = [
            (e + s, ind)
            for ind, e, s in zip(candidates, entropy_scores, synergy_scores)
        ]
        combined_scores.sort(key=lambda x: x[0], reverse=True)

        # Step 3: Dynamic Modulation of Selection Pressure
        selected.extend(ind for _, ind in combined_scores[:2] if len(selected) < k)

    return selected[:k]


def diverse_performance_selection(population, k=1):
    def adaptive_lexicase_filter(individuals, num_cases):
        """Performs adaptive lexicase filtering."""
        candidates = individuals[:]
        case_indices = np.random.permutation(num_cases)
        for idx in case_indices:
            min_error = min(ind.case_values[idx] for ind in candidates)
            candidates = [
                ind for ind in candidates if ind.case_values[idx] <= min_error
            ]
            if len(candidates) <= k:
                break
        return candidates

    def entropy_diversity(ind):
        """Calculate entropy-based diversity score."""
        values = np.array(ind.case_values)
        probabilities = values / np.sum(values)
        return -np.sum(probabilities * np.log(probabilities + 1e-9))

    def synergy_metric(candidates):
        """Calculate synergy based on pairwise performance agreement."""
        error_matrix = np.array([ind.case_values for ind in candidates])
        inverse_error_agreement = 1 - np.abs(np.corrcoef(error_matrix))
        synergy_scores = np.sum(inverse_error_agreement, axis=0)
        return synergy_scores

    num_cases = len(population[0].case_values)
    selected_individuals = []

    while len(selected_individuals) < k:
        # Adaptive lexicase filtering
        candidates = adaptive_lexicase_filter(population, num_cases)

        if len(candidates) == 1:
            selected_individuals.append(candidates[0])
            continue

        # Diversity selection using entropy
        diversity_scores = [entropy_diversity(ind) for ind in candidates]
        best_diverse_candidate = candidates[np.argmax(diversity_scores)]

        # Synergy-based candidate selection
        synergy_scores = synergy_metric(candidates)
        best_synergy_idx = np.argmax(synergy_scores)
        selected_pair = [best_diverse_candidate, candidates[best_synergy_idx]]

        for ind in selected_pair:
            if len(selected_individuals) < k:
                selected_individuals.append(ind)

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
