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


def decay_rank_selection(population, k=1):
    # Calculate mean error for each individual
    mean_errors = np.array([np.mean(ind.case_values) for ind in population])

    # Rank individuals by mean error (lower is better)
    ranked_indices = np.argsort(mean_errors)
    ranked_population = [population[i] for i in ranked_indices]

    # Use exponential decay on rank to assign probabilities
    rank_weights = np.exp(-np.arange(len(population)))
    total_weight = np.sum(rank_weights)
    probabilities = rank_weights / total_weight

    # Perform selection using these probabilities
    selected_indices = np.random.choice(
        len(population), size=k, replace=False, p=probabilities
    )
    selected_individuals = [ranked_population[i] for i in selected_indices]

    return selected_individuals


def weighted_decay_rank_selection(population, k=1):
    # Calculate mean error for each individual
    mean_errors = np.array([np.mean(ind.case_values) for ind in population])

    # Rank individuals by mean error (lower is better)
    ranked_indices = np.argsort(mean_errors)
    ranked_population = [population[i] for i in ranked_indices]
    ranked_mean_errors = mean_errors[ranked_indices]

    # Calculate fitness scores as inverse of errors, scaled by max ranked error
    max_ranked_error = ranked_mean_errors[-1]
    fitness_scores = [
        (max_ranked_error - error + 1) for error in ranked_mean_errors
    ]  # Avoid zero fitness

    # Calculate rank-based weights
    rank_weights = np.exp(-np.arange(len(ranked_population)))

    # Combine rank weights and fitness scores
    combined_weights = np.array(fitness_scores) * rank_weights

    # Normalize combined weights to form a probability distribution
    total_weight = np.sum(combined_weights)

    # Handle the edge case of all zero weights
    if total_weight == 0:
        return random.sample(population, k)

    probabilities = combined_weights / total_weight

    # Select individuals based on these combined probabilities
    selected_indices = np.random.choice(
        len(ranked_population), size=k, replace=False, p=probabilities
    )
    selected_individuals = [ranked_population[i] for i in selected_indices]

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
