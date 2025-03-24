import random

import numpy as np
from deap import tools
from scipy.stats import pearsonr

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast


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


def select_cpsr_regression(population, k, target, metric="MSE"):
    """
    CPS selection (residual version) for symbolic regression with vectorized NumPy operations.

    Assumes:
    - individual.case_values is a NumPy array of squared errors per case
    """
    selected = []

    for _ in range(k):
        # Select the first parent using epsilon-lexicase selection
        mother = selAutomaticEpsilonLexicaseFast(population, 1)[0]
        M_pred = mother.predicted_values

        best_father = None
        best_fitness = -float("inf")

        for father in population:
            if father is mother:
                continue

            F_pred = father.predicted_values
            avg_pred = (M_pred + F_pred) / 2

            # Calculate fitness based on selected metric
            if metric == "MSE":
                fit = -np.mean((avg_pred - target) ** 2)
            elif metric == "Pearson":
                if np.std(avg_pred) == 0 or np.std(target) == 0:
                    fit = -1  # Avoid division by zero, treat as worst correlation
                else:
                    fit, _ = pearsonr(avg_pred, target)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            # Select best father
            if fit > best_fitness:
                best_father = father
                best_fitness = fit
            elif fit == best_fitness:
                if np.sum(father.case_values) < np.sum(best_father.case_values):
                    best_father = father

        selected.extend([mother, best_father])

    return selected


def select_cps_regression(population, k, operator="Tournament"):
    """
    CPS selection for symbolic regression with vectorized NumPy operations.

    Assumes:
    - individual.case_values is a NumPy array of squared errors per case
    """
    selected = []

    for _ in range(k):
        # 1. Tournament selection for mother (tournsize=3)
        if operator == "Tournament":
            mother = tools.selTournament(population, 1, tournsize=3)[0]
        else:
            mother = selAutomaticEpsilonLexicaseFast(population, 1)[0]
        M = mother.case_values  # NumPy array

        best_father = None
        best_fitness = -float("inf")

        for father in population:
            if father is mother:
                continue
            F = father.case_values  # NumPy array

            # 2. Vectorized best-case offspring errors
            BF_M = np.minimum(F, M)  # Element-wise min
            fit = -np.mean(BF_M)  # Fitness = negative mean error

            if fit > best_fitness:
                best_father = father
                best_fitness = fit
            elif fit == best_fitness:
                if np.sum(F) < np.sum(best_father.case_values):
                    best_father = father

        selected.extend([mother, best_father])

    return selected


def complementary_tournament(population, k=100, tour_size=3):
    if not population:
        return []

    num_cases = population[0].case_values.size
    pop_size = len(population)

    # Compute average ranks per individual across all cases
    case_ranks = np.zeros((pop_size, num_cases))
    for case_idx in range(num_cases):
        case_errors = np.array([ind.case_values[case_idx] for ind in population])
        ranks = np.argsort(np.argsort(case_errors)) + 1
        case_ranks[:, case_idx] = ranks
    avg_ranks = np.mean(case_ranks, axis=1)

    def tournament_selection(fitness_values, individuals):
        competitor_indices = random.sample(range(len(individuals)), tour_size)
        competitors = [individuals[i] for i in competitor_indices]
        competitor_fitnesses = [fitness_values[i] for i in competitor_indices]
        best_idx = min(range(tour_size), key=lambda i: competitor_fitnesses[i])
        return competitors[best_idx]

    selected_individuals = []

    for _ in range(k // 2):
        # Select Parent A using average rank
        parent_a = tournament_selection(avg_ranks, population)
        selected_individuals.append(parent_a)

        parent_a_index = population.index(parent_a)
        parent_a_errors = population[parent_a_index].case_values
        error_threshold = np.percentile(parent_a_errors, 75)
        weak_case_indices = np.where(parent_a_errors >= error_threshold)[0]

        if weak_case_indices.size > 0:
            comp_scores = np.array(
                [np.mean(ind.case_values[weak_case_indices])
                 for ind in population]
            )
            valid_indices = [i for i in range(pop_size)
                             if i != parent_a_index]
            parent_b = tournament_selection(
                comp_scores[valid_indices],
                [population[i] for i in valid_indices]
            )
            selected_individuals.append(parent_b)

    return selected_individuals


def novel_selection(population, k=100, status=None, tour_size=3):
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


def novel_selection_plus(
    population,
    k=1,
    tour_size=3,
    complexity_penalty_lambda=0.01,
    complementarity_weight=1.0,
    diversity_penalty_lambda=0.1,
):
    if not population:
        return []

    num_cases = population[0].case_values.size
    pop_size = len(population)
    selected_individuals = []

    # 1. Calculate Case Ranks and Average Ranks (vectorized) - as in Operator B and C
    case_ranks = np.zeros((pop_size, num_cases))
    for case_idx in range(num_cases):
        case_errors = np.array([ind.case_values[case_idx] for ind in population])
        ranked_indices = np.argsort(case_errors)
        ranks = np.argsort(ranked_indices) + 1
        case_ranks[:, case_idx] = ranks

    avg_ranks = np.mean(case_ranks, axis=1)

    # 2. Define Fitness Function incorporating complexity penalty - as in Operator C
    def calculate_fitness(individual, avg_ranks_vector):
        ind_index = population.index(individual)
        avg_rank = avg_ranks_vector[ind_index]
        complexity_score = (
            len(individual) + individual.height
        )  # Example complexity: nodes + height
        fitness_value = (
            avg_rank + complexity_penalty_lambda * complexity_score
        )  # Penalize complexity
        return fitness_value

    def tournament_selection(
        fitness_values, individuals, tournament_k, minimize=True, score_array=None
    ):
        """Helper function for standard tournament selection using pre-calculated fitness, can use score_array for custom scores."""
        selected_tournament = []
        indices = list(range(len(individuals)))
        for _ in range(tournament_k):
            competitor_indices = random.sample(indices, tour_size)
            competitors = [individuals[i] for i in competitor_indices]
            if score_array is None:
                competitor_fitnesses = [fitness_values[i] for i in competitor_indices]
            else:
                competitor_fitnesses = [
                    score_array[i] for i in competitor_indices
                ]  # Use custom scores if provided

            if minimize:
                best_idx = min(range(tour_size), key=lambda i: competitor_fitnesses[i])
            else:  # maximize (if needed in future)
                best_idx = max(range(tour_size), key=lambda i: competitor_fitnesses[i])

            winner = competitors[best_idx]
            selected_tournament.append(winner)

        if score_array is None:
            best_individual = min(
                selected_tournament,
                key=lambda ind: fitness_values[individuals.index(ind)],
            )  # Select best from tournament based on fitness
        else:
            best_individual = min(
                selected_tournament, key=lambda ind: score_array[individuals.index(ind)]
            )  # Select best from tournament based on custom scores

        return best_individual

    fitness_values = [
        calculate_fitness(ind, avg_ranks) for ind in population
    ]  # Calculate fitness for all individuals once

    for _ in range(k // 2):  # Select pairs of parents
        # 3. Parent A Selection: Fitness-based tournament (incorporating complexity) - as in Operator C
        parent_a = tournament_selection(
            fitness_values, population, tournament_k=tour_size, minimize=True
        )
        selected_individuals.append(parent_a)

        # 4. Parent B Selection: Novel Complementarity and Diversity Focused Tournament
        parent_a_index = population.index(parent_a)
        parent_a_errors = parent_a.case_values

        complementarity_diversity_scores = np.zeros(pop_size)
        for ind_idx in range(pop_size):
            if ind_idx == parent_a_index:  # Exclude Parent A from Parent B selection
                complementarity_diversity_scores[ind_idx] = float(
                    "inf"
                )  # Effectively exclude by making score very high (for minimization tournament)
                continue

            ind_b = population[ind_idx]
            ind_b_errors = ind_b.case_values

            # Complementarity Score: Negative correlation of errors
            error_correlation = np.corrcoef(parent_a_errors, ind_b_errors)[0, 1]
            complementarity_score = (
                -error_correlation if not np.isnan(error_correlation) else 0.0
            )  # Maximize negative correlation

            # Diversity Penalty: Similarity of error profiles (Euclidean distance - higher distance is better for diversity)
            error_distance = np.linalg.norm(parent_a_errors - ind_b_errors)
            diversity_penalty = (
                -error_distance * diversity_penalty_lambda
            )  # Penalize similarity (minimize negative distance)

            # Combine Complementarity and Diversity (weighted sum)
            complementarity_diversity_scores[ind_idx] = (
                -complementarity_weight * complementarity_score + diversity_penalty
            )  # Minimize this score

        # Tournament selection for Parent B using the combined complementarity and diversity score
        valid_indices_for_b = [
            i for i in range(pop_size) if i != parent_a_index
        ]  # Already handled in score calculation, but for clarity
        if not valid_indices_for_b:
            parent_b = tournament_selection(
                fitness_values, population, tournament_k=tour_size, minimize=True
            )  # Fallback: fitness-based if no other options
        else:
            valid_population_for_b = [population[i] for i in valid_indices_for_b]
            valid_complementarity_diversity_scores = complementarity_diversity_scores[
                valid_indices_for_b
            ]
            parent_b = tournament_selection(
                list(valid_complementarity_diversity_scores),
                valid_population_for_b,
                tournament_k=tour_size,
                minimize=True,
                score_array=valid_complementarity_diversity_scores,
            )

        selected_individuals.append(parent_b)

    # If k is odd, add one more (e.g., best overall fitness) - as in Operator C
    while len(selected_individuals) < k:
        selected_individuals.append(
            tournament_selection(
                fitness_values, population, tournament_k=tour_size, minimize=True
            )
        )

    return selected_individuals


def novel_selection_plus_plus(population, k=1):
    return novel_selection(population, k, 3)


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
