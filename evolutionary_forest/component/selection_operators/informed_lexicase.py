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
    def get_information(individual):
        mse_vector = individual.case_values
        predicted_values = individual.predicted_values
        residual = individual.y - predicted_values
        number_of_nodes = len(individual)
        height = individual.height
        return mse_vector, predicted_values, residual, number_of_nodes, height

    def calculate_fitness_and_diversity(selection):
        # Mean Squared Error for fitness (lower is better)
        fitness_scores = [np.mean(ind.case_values) for ind in selection]
        # Calculate diversity using pairwise distances
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

    def select_parent(selection, fitness_scores, diversity_scores, other_parent=None):
        # Sort candidates based on fitness (ascending)
        sorted_indices = np.argsort(fitness_scores)
        best_candidate_idx = sorted_indices[0]

        # Optimize on compatibility with the existing parent
        if other_parent is not None:
            for idx in sorted_indices:
                candidate = selection[idx]
                compatibility_score = np.linalg.norm(
                    candidate.case_values - other_parent.case_values
                )
                if compatibility_score >= np.median(
                    diversity_scores[best_candidate_idx]
                ):
                    return candidate
            return selection[best_candidate_idx]  # Fallback to best candidate

        return selection[best_candidate_idx]

    selected_individuals = []

    while len(selected_individuals) < k:
        # Select for Parent A
        tournament_a = random.sample(population, tour_size)
        fitness_a, diversity_a = calculate_fitness_and_diversity(tournament_a)
        parent_a = select_parent(tournament_a, fitness_a, diversity_a)
        selected_individuals.append(parent_a)

        # Select for Parent B
        tournament_b = random.sample(population, tour_size)
        fitness_b, diversity_b = calculate_fitness_and_diversity(tournament_b)
        parent_b = select_parent(
            tournament_b, fitness_b, diversity_b, other_parent=parent_a
        )

        selected_individuals.append(parent_b)

    # Trim the list of selected individuals to the requested size
    return selected_individuals[:k]


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
