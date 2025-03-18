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
    if not population or k <= 0:
        return []

    # Extract case_values into a 2D numpy array
    case_matrix = np.array([ind.case_values for ind in population])  # Shape: (n, m)
    n_individuals, n_cases = case_matrix.shape

    # Compute fitness as inverse of mean error across cases
    epsilon = 1e-8
    mean_errors = case_matrix.mean(axis=1)  # Shape: (n,)
    fitness = 1.0 / (mean_errors + epsilon)  # Higher fitness is better

    # Normalize fitness
    fitness_normalized = fitness / fitness.sum()

    # Adaptive Fitness Sharing to maintain diversity
    # Compute similarity matrix using cosine similarity
    norms = np.linalg.norm(case_matrix, axis=1, keepdims=True) + epsilon
    normalized_cases = case_matrix / norms
    similarity_matrix = np.dot(normalized_cases, normalized_cases.T)  # Shape: (n, n)

    # Define similarity threshold for sharing
    similarity_threshold = 0.7  # Adjust as needed

    # Fitness Sharing: Reduce fitness based on similarity
    sharing_factors = (similarity_matrix > similarity_threshold).sum(axis=1)
    sharing_factors = np.maximum(sharing_factors, 1)  # Avoid division by zero
    fitness_shared = fitness / sharing_factors

    # Normalize shared fitness for selection probabilities
    fitness_shared_normalized = fitness_shared / fitness_shared.sum()

    # Selection Probability proportional to shared fitness
    selection_probs = fitness_shared_normalized

    # Select k individuals based on selection_probs
    selected_indices = np.random.choice(
        n_individuals, size=k, replace=True, p=selection_probs
    )
    selected_individuals = [population[idx] for idx in selected_indices]

    # Crossover Compatibility: Pair selected individuals intelligently
    # Compute pairwise similarity among selected individuals
    selected_matrix = case_matrix[selected_indices]  # Shape: (k, m)
    selected_norms = np.linalg.norm(selected_matrix, axis=1, keepdims=True) + epsilon
    selected_normalized = selected_matrix / selected_norms
    pair_similarity = np.dot(
        selected_normalized, selected_normalized.T
    )  # Shape: (k, k)

    # Avoid self-pairing by setting diagonal to -inf
    np.fill_diagonal(pair_similarity, -np.inf)

    # Assign compatibility scores (higher similarity implies higher compatibility)
    # Here, we prefer moderate similarity to encourage diversity among parents
    # You can adjust this logic based on specific requirements
    compatibility_scores = (
        1 - pair_similarity
    )  # Lower similarity => higher compatibility

    # Create a list to track paired individuals
    paired = set()
    final_selected = []

    for i in range(k):
        if i in paired:
            continue
        # Find the most compatible individual to pair with
        compatibilities = compatibility_scores[i]
        # Exclude already paired individuals
        compatibilities[list(paired)] = np.inf
        partner = np.argmin(compatibilities)
        if compatibilities[partner] == np.inf:
            # No available partner, add as is
            final_selected.append(selected_individuals[i])
            paired.add(i)
        else:
            # Add both individuals as a compatible pair
            final_selected.extend(
                [selected_individuals[i], selected_individuals[partner]]
            )
            paired.update([i, partner])
        if len(final_selected) >= k:
            break

    # In case of odd k, trim the list
    selected_individuals = final_selected[:k]

    return selected_individuals


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
