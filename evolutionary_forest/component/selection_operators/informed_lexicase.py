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
    if not population:
        return []

    num_cases = population[0].case_values.size
    pop_size = len(population)
    selected_individuals = []

    # 1. Calculate Case Ranks and Average Ranks (vectorized)
    case_ranks = np.zeros((pop_size, num_cases))
    for case_idx in range(num_cases):
        case_errors = np.array([ind.case_values[case_idx] for ind in population])
        ranked_indices = np.argsort(
            case_errors
        )  # Indices of individuals sorted by error for this case
        ranks = np.argsort(ranked_indices) + 1  # Convert indices to ranks (1-based)
        case_ranks[:, case_idx] = ranks

    avg_ranks = np.mean(case_ranks, axis=1)

    def tournament_selection(
        fitness_values,
        individuals,
        tournament_k,
        minimize=True,
        tie_breaker_complexity=True,
    ):
        """Helper function for tournament selection."""
        selected_tournament = []
        indices = list(range(len(individuals)))
        for _ in range(tournament_k):
            competitor_indices = random.sample(indices, tour_size)
            competitors = [individuals[i] for i in competitor_indices]
            competitor_fitnesses = [fitness_values[i] for i in competitor_indices]

            if minimize:
                best_idx = min(range(tour_size), key=lambda i: competitor_fitnesses[i])
            else:  # maximize
                best_idx = max(range(tour_size), key=lambda i: competitor_fitnesses[i])

            winner = competitors[best_idx]
            selected_tournament.append(winner)

        # Tie-breaker using complexity (nodes then height) if requested
        if tie_breaker_complexity:

            def complexity_score(ind):
                return (len(ind), ind.height)  # Nodes first, then height

            best_individual = min(
                selected_tournament,
                key=lambda ind: (
                    fitness_values[individuals.index(ind)],
                    complexity_score(ind),
                ),
            )
        else:
            if minimize:
                best_individual = min(
                    selected_tournament,
                    key=lambda ind: fitness_values[individuals.index(ind)],
                )
            else:
                best_individual = max(
                    selected_tournament,
                    key=lambda ind: fitness_values[individuals.index(ind)],
                )

        return best_individual

    for _ in range(k // 2):  # Select pairs of parents
        # 2. Parent A Selection: Rank-based tournament
        parent_a = tournament_selection(
            avg_ranks, population, tournament_k=tour_size, minimize=True
        )
        selected_individuals.append(parent_a)

        # 3. Parent B Selection: Complementarity-focused tournament
        parent_a_index = population.index(parent_a)
        parent_a_errors = population[parent_a_index].case_values

        # Identify cases where Parent A performs relatively poorly (top quartile of errors)
        error_threshold_a = np.percentile(
            parent_a_errors, 75
        )  # Cases with errors in top 25%
        poor_cases_a_indices = np.where(parent_a_errors >= error_threshold_a)[0]

        if poor_cases_a_indices.size > 0:  # If there are cases where parent A is weak
            complementarity_scores = np.zeros(pop_size)
            for ind_idx in range(pop_size):
                complementarity_scores[ind_idx] = np.mean(
                    population[ind_idx].case_values[poor_cases_a_indices]
                )

            # Exclude parent_a from selection pool for parent_b to encourage diversity in pairing
            valid_indices_for_b = [i for i in range(pop_size) if i != parent_a_index]
            if not valid_indices_for_b:  # Handle case where population size is very small or only one unique individual left
                parent_b = tournament_selection(
                    avg_ranks, population, tournament_k=tour_size, minimize=True
                )  # Fallback: rank based selection
            else:
                valid_population_for_b = [population[i] for i in valid_indices_for_b]
                valid_complementarity_scores = complementarity_scores[
                    valid_indices_for_b
                ]
                parent_b = tournament_selection(
                    valid_complementarity_scores,
                    valid_population_for_b,
                    tournament_k=tour_size,
                    minimize=True,
                )

        else:  # If parent_a is strong across all cases, fallback to rank-based selection for parent_b for diversity
            parent_b = tournament_selection(
                avg_ranks, population, tournament_k=tour_size, minimize=True
            )

        selected_individuals.append(parent_b)

    # If k is odd, we might have selected k-1, add one more (e.g., best overall rank)
    while len(selected_individuals) < k:
        selected_individuals.append(
            tournament_selection(
                avg_ranks, population, tournament_k=tour_size, minimize=True
            )
        )

    return selected_individuals


def llm_selection_plus(population, k=1, tour_size=7):
    def get_information(individual):
        mse_vector = individual.case_values
        predicted_values = individual.predicted_values
        residual = individual.y - predicted_values
        number_of_nodes = len(individual)
        height = individual.height
        return mse_vector, predicted_values, residual, number_of_nodes, height

    def calculate_fitness_and_diversity(selection):
        fitness_scores = [
            np.mean(ind.case_values) for ind in selection
        ]  # Mean error (lower is better)
        diversity_matrix = np.array(
            [
                [
                    np.linalg.norm(ind_a.case_values - ind_b.case_values)
                    for ind_b in selection
                ]
                for ind_a in selection
            ]
        )
        return fitness_scores, diversity_matrix

    def rank_selection(selection):
        fitness_scores, _ = calculate_fitness_and_diversity(selection)
        ranked_indices = np.argsort(
            fitness_scores
        )  # Sort indices of the population by fitness (lower is better)
        return selection[ranked_indices[0]]  # Return the best individual

    def select_with_compatibility(tournament, other_parent):
        fitness_scores, diversity_scores = calculate_fitness_and_diversity(tournament)
        best_idx = np.argmin(fitness_scores)
        best_candidate = tournament[best_idx]

        # Check compatibility with the other parent (retaining diversity)
        compatibility_indices = np.where(
            diversity_scores[best_idx] > np.percentile(diversity_scores, 50)
        )[0]

        if len(compatibility_indices) > 0:
            compatible_candidates = [tournament[i] for i in compatibility_indices]
            compatible_fitness = [fitness_scores[i] for i in compatibility_indices]
            best_compatible_idx = np.argmin(compatible_fitness)
            return compatible_candidates[best_compatible_idx]

        return best_candidate  # Fallback to the best if no compatibility is found

    selected_individuals = []

    for _ in range(k // 2):
        # Select Parent A
        tournament_a = random.sample(population, tour_size)
        parent_a = rank_selection(tournament_a)
        selected_individuals.append(parent_a)

        # Select Parent B considering compatibility with Parent A
        tournament_b = random.sample(population, tour_size)
        parent_b = select_with_compatibility(tournament_b, parent_a)
        selected_individuals.append(parent_b)

    return selected_individuals[:k]


def llm_selection_plus_plus(population, k=1, tour_size=7):
    pass


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
