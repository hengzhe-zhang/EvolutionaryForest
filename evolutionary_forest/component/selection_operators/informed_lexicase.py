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
        return mse_vector, predicted_values, residual, number_of_nodes

    def evaluate(individual):
        mse_vector, predicted_values, residual, number_of_nodes = get_information(
            individual
        )
        mse = np.mean(mse_vector)  # Lower MSE is better
        return mse, predicted_values, residual, number_of_nodes

    def compute_compatibility(parent_a, candidate):
        mse_a = np.mean(parent_a.case_values)
        mse_b = np.mean(candidate.case_values)
        return (
            abs(mse_a - mse_b) < 0.1 and abs(len(parent_a) - len(candidate)) <= 2
        )  # MSE and structure proximity

    def tournament_selection(tournament):
        fitness_scores = [(ind, evaluate(ind)) for ind in tournament]
        fitness_scores.sort(
            key=lambda item: item[1][0]
        )  # Sort by MSE (lower is better)
        selected = fitness_scores[0][0]  # Select the best individual
        # Adaptively adjust selection based on the top few candidates to maintain diversity
        if len(fitness_scores) > 1:
            diversity_factor = random.uniform(0, 1)
            if diversity_factor < 0.5:  # Encouraging variability randomly
                selected = random.choice([ind for ind, _ in fitness_scores[:3]])
        return selected

    selected_individuals = []

    for _ in range(k // 2):
        # Step 1: Select parent_a from a larger tournament
        tournament_a = random.sample(population, tour_size)
        parent_a = tournament_selection(tournament_a)
        selected_individuals.append(parent_a)

        # Step 2: Select a compatible parent_b
        tournament_b = random.sample(population, tour_size)
        compatible_individuals = [
            ind for ind in tournament_b if compute_compatibility(parent_a, ind)
        ]

        if compatible_individuals:  # Favor compatible individuals
            parent_b = random.choice(compatible_individuals)
        else:  # If none found, select the best one from tournament_b
            parent_b = tournament_selection(tournament_b)

        selected_individuals.append(parent_b)

    return selected_individuals


def llm_selection_plus(population, k=1, tour_size=7):
    def get_information(individual):
        # Extract metrics from the individual
        mse_vector = individual.case_values
        predicted_values = individual.predicted_values
        residual = individual.y - predicted_values
        number_of_nodes = (
            individual.number_of_nodes if hasattr(individual, "number_of_nodes") else 1
        )
        height = individual.height if hasattr(individual, "height") else 1
        return mse_vector, predicted_values, residual, number_of_nodes, height

    def calculate_fitness(individual):
        mse_vector, _, _, number_of_nodes, height = get_information(individual)
        # Using mean squared error as fitness (lower is better)
        fitness = np.mean(mse_vector)
        return fitness, number_of_nodes, height

    def tournament_selection(population, tour_size):
        tournament_individuals = random.sample(population, tour_size)
        ranked_individuals = sorted(
            tournament_individuals, key=lambda ind: calculate_fitness(ind)
        )
        return ranked_individuals

    def select_compatible_parent(rank_a, rank_b):
        # Select the best individual while ensuring they are compatible
        parent_a = rank_a[0]  # Best individual from the first tournament
        compatible_candidates = [ind for ind in rank_b if ind != parent_a]

        if compatible_candidates:
            parent_b = min(
                compatible_candidates, key=lambda ind: calculate_fitness(ind)[0]
            )
        else:
            parent_b = random.choice(rank_b)  # fallback if no compatible candidates

        return parent_a, parent_b

    selected_individuals = []

    # Main selection process
    while len(selected_individuals) < k:
        # Perform tournament selections to get two ranked sets
        rank_a = tournament_selection(population, tour_size)
        rank_b = tournament_selection(population, tour_size)

        # Select a pair of parents with compatibility in mind
        parent_a, parent_b = select_compatible_parent(rank_a, rank_b)

        selected_individuals.append(parent_a)
        selected_individuals.append(parent_b)

    return selected_individuals[:k]  # Ensure we only return k individuals


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
