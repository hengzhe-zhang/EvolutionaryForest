import random
import numpy as np
from sklearn.cluster import KMeans


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


def llm_selection(population, k=100, num_clusters=5):
    """
    Innovative selection operator for genetic programming focusing on diversity,
    crossover compatibility, and efficiency.
    """

    def normalize_case_values(ind):
        """Normalize an individual's case values for fair comparison across individuals."""
        case_min = np.min(ind.case_values)
        case_max = np.max(ind.case_values)
        return (ind.case_values - case_min) / (case_max - case_min)

    # Step 1: Normalize the case values
    normalized_population = np.array([normalize_case_values(ind) for ind in population])

    # Step 2: Apply clustering to categorize individuals into groups
    kmeans = KMeans(n_clusters=min(num_clusters, len(population)), random_state=0)
    cluster_labels = kmeans.fit_predict(normalized_population)

    # Step 3: Select representative individuals from each cluster for diversity
    cluster_representatives = []
    for cluster_id in range(num_clusters):
        cluster_members = [
            ind for ind, label in zip(population, cluster_labels) if label == cluster_id
        ]
        if cluster_members:
            # Choose the individual with lowest mean error in the cluster to represent it
            best_member = min(cluster_members, key=lambda ind: np.mean(ind.case_values))
            cluster_representatives.append(best_member)

    # Ensure there's enough candidates to select pairs
    if len(cluster_representatives) < 2:
        cluster_representatives = population[:]

    selected_individuals = []
    while len(selected_individuals) < k:
        parent1 = random.choice(cluster_representatives)

        # Calculate compatibility scores with other representatives
        compatibility_scores = [
            (
                np.linalg.norm(
                    normalize_case_values(parent1) - normalize_case_values(ind)
                ),
                ind,
            )
            for ind in cluster_representatives
            if ind != parent1
        ]

        # Select the most compatible partner
        compatible_partner = min(compatibility_scores, key=lambda x: x[0])[1]

        # Add to the selection if space is available
        selected_individuals.append(parent1)
        if len(selected_individuals) < k:
            selected_individuals.append(compatible_partner)

    return selected_individuals[:k]


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
