import numpy as np
import random


def rds_tournament_selection(population, k=100, tour_size=7):
    def downsample_indices(total_cases, fraction=0.1):
        num_samples = max(1, int(total_cases * fraction))
        return random.sample(range(total_cases), num_samples)

    total_cases = len(population[0].case_values)  # Number of training cases

    selected = []

    for _ in range(k):
        tournament = random.sample(population, tour_size)
        downsampled_indices = downsample_indices(total_cases)

        # Compute temporary fitness using the down-sampled indices
        temp_fitness = {}
        for ind in tournament:
            sampled_errors = [ind.case_values[i] for i in downsampled_indices]
            temp_fitness[ind] = np.mean(sampled_errors)

        # Select the best individual based on temporary fitness
        winner = min(tournament, key=lambda ind: temp_fitness[ind])
        selected.append(winner)

    return selected
