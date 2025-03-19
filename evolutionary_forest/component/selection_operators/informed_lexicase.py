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
        # Select parent B as the most compatible with parent A
        parent_b = min(tournament_b, key=lambda ind: compatibility_score(parent_a, ind))
        selected_individuals.append(parent_b)

    return selected_individuals


def llm_selection_plus(population, k=1, tour_size=7):
    def get_information(individual):
        mse_vector = np.array(individual.case_values)
        predicted_values = np.array(individual.predicted_values)
        residual = individual.y - predicted_values
        number_of_nodes = len(individual.predicted_values)
        return mse_vector, predicted_values, residual, number_of_nodes

    # Specialized score that encourages diverse and strong individuals
    def specialized_score(ind, diversity_weight=0.3):
        mse, _, _, nodes = get_information(ind)
        diversity_bonus = diversity_weight / (1 + nodes)
        return mse.mean() + diversity_bonus

    # Score for selecting highly compatible parents for crossover
    def compatibility_score(ind1, ind2):
        mse1, _, residual1, _ = get_information(ind1)
        mse2, _, residual2, _ = get_information(ind2)
        mse_similarity = np.abs(mse1.mean() - mse2.mean())
        residual_alignment = np.dot(residual1, residual2)
        return mse_similarity + residual_alignment

    selected_individuals = []

    # Precompute diversity factor for performance benefit
    diversity_values = [specialized_score(ind) for ind in population]
    diversity_threshold = np.median(diversity_values)

    for _ in range(k // 2):
        # Pick parent A ensuring diversity and specialization
        tournament_a = random.sample(population, tour_size)
        # Rank parents with specialty ensuring it's above a diversity threshold
        parent_a_candidates = [
            ind for ind in tournament_a if specialized_score(ind) < diversity_threshold
        ]
        if parent_a_candidates:
            parent_a = min(parent_a_candidates, key=specialized_score)
        else:
            parent_a = min(tournament_a, key=specialized_score)
        selected_individuals.append(parent_a)

        # Select parent B with enhanced compatibility to parent A
        tournament_b = random.sample(population, tour_size)
        compatibility_scores = [
            compatibility_score(parent_a, ind) for ind in tournament_b
        ]
        diversity_scores = [specialized_score(ind) for ind in tournament_b]

        # Rank candidates based on a combination of compatibility and diversity
        combined_scores = [
            (0.7 * comp + 0.3 * div)
            for comp, div in zip(compatibility_scores, diversity_scores)
        ]
        min_index = combined_scores.index(min(combined_scores))
        parent_b = tournament_b[min_index]
        selected_individuals.append(parent_b)

    return selected_individuals


def llm_selection_plus_plus(population, k=1, tour_size=7):
    def get_information(individual):
        mse_vector = individual.case_values
        predicted_values = individual.predicted_values
        residual = individual.y - individual.predicted_values
        number_of_nodes = len(individual)
        return mse_vector, predicted_values, residual, number_of_nodes

    def composite_score(ind):
        mse, _, _, nodes = get_information(ind)
        return mse.mean() + 0.1 * nodes  # Penalize larger trees slightly

    def diversity_score(predicted_values):
        return len(set(predicted_values))

    def compatibility_score(ind1, ind2):
        _, _, residual1, _ = get_information(ind1)
        _, _, residual2, _ = get_information(ind2)
        return np.dot(residual1, residual2)

    selected_individuals = []

    while len(selected_individuals) < k:
        potential_parents = []

        # Select tour_size individuals
        for _ in range(tour_size):
            candidate = random.choice(population)
            mse_vector, predicted_values, residual, num_nodes = get_information(
                candidate
            )
            fitness = 1.0 / np.mean(mse_vector)  # Fitness as inverse of MSE
            diversity = diversity_score(predicted_values)
            potential_parents.append((candidate, fitness, diversity))

        # Sort candidates based on fitness and diversity
        potential_parents.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Select top ranked based on combined fitness and diversity
        parent_a = potential_parents[0][0]
        selected_individuals.append(parent_a)

        # Select a compatible partner for parent_a
        best_compatibility = float("inf")
        for candidate, _, _ in potential_parents[1:]:
            comp_score = compatibility_score(parent_a, candidate)
            if comp_score < best_compatibility:
                best_compatibility = comp_score
                parent_b = candidate

        selected_individuals.append(parent_b)

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
