import random

import numpy as np
from deap import tools
from scipy.stats import spearmanr

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast


def fast_pearson(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


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
                    fit = fast_pearson(avg_pred, target)
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
                [np.mean(ind.case_values[weak_case_indices]) for ind in population]
            )
            valid_indices = [i for i in range(pop_size) if i != parent_a_index]
            parent_b = tournament_selection(
                comp_scores[valid_indices], [population[i] for i in valid_indices]
            )
            selected_individuals.append(parent_b)

    return selected_individuals


def novel_selection(population, k=100, status={}):
    """
    A novel selection operator for genetic programming that combines aspects of tournament selection,
    epsilon lexicase selection, and Pareto optimality to balance exploitation, exploration, and
    parsimony pressure.  It focuses on improved complementarity by using Spearman correlation
    instead of raw MSE reduction when selecting parent_b.
    """
    pop_size = len(population)
    if pop_size == 0:
        return []

    if k > pop_size:
        k = pop_size

    if k % 2 != 0:
        k -= 1  # Ensure k is even for pairing

    evolutionary_stage = status.get("evolutionary_stage", 0.0)  # Default to early stage

    # 1. Fitness Evaluation and Normalization
    mse_vectors = np.array([ind.case_values for ind in population])
    individual_lengths = np.array([len(ind) for ind in population])
    individual_heights = np.array([ind.height for ind in population])
    residuals = np.array([ind.y - ind.predicted_values for ind in population])

    # Overall fitness (mean MSE)
    fitness_values = np.mean(mse_vectors, axis=1)

    # Normalize fitness, length, and height for combined scoring
    fitness_norm = (fitness_values - np.min(fitness_values)) / (
        np.max(fitness_values) - np.min(fitness_values) + 1e-8
    )
    length_norm = (individual_lengths - np.min(individual_lengths)) / (
        np.max(individual_lengths) - np.min(individual_lengths) + 1e-8
    )
    height_norm = (individual_heights - np.min(individual_heights)) / (
        np.max(individual_heights) - np.min(individual_heights) + 1e-8
    )

    # Stage-dependent weights:  Early stage favors exploration (diversity), late stage favors exploitation (fitness) and interpretability.
    fitness_weight = (
        0.3 + 0.7 * evolutionary_stage
    )  # Increased fitness as evolution continues
    length_weight = (
        0.2 - 0.2 * evolutionary_stage
    )  # Reduce tree size as evolution progresses.
    height_weight = 0.2 - 0.2 * evolutionary_stage  # reduce height
    diversity_weight = (
        0.5 - 0.4 * evolutionary_stage
    )  # Diversity decreases as evolution proceeds, allowing for convergence

    # Calculate combined score
    combined_score = (
        fitness_weight
        * (1 - fitness_norm)  # Invert fitness_norm because lower is better
        + length_weight * (1 - length_norm)  # Prefer shorter lengths
        + height_weight * (1 - height_norm)
    )  # Prefer lower heights

    # 2. Diversity Promotion using Epsilon Lexicase
    def epsilon_lexicase_selection(individuals, epsilon=0.01):
        """Selects an individual using epsilon-lexicase selection."""
        num_cases = len(individuals[0].case_values)

        # Start with all individuals as candidates
        candidates = list(range(len(individuals)))

        for case in range(num_cases):
            if len(candidates) == 1:
                break

            # Find the best performance on this case
            case_errors = np.array(
                [individuals[i].case_values[case] for i in candidates]
            )
            best_error = np.min(case_errors)

            # Identify individuals within epsilon of the best error
            nearly_best = case_errors <= (best_error + epsilon)

            # Reduce the candidate pool
            candidates = [
                candidates[i]
                for i, is_nearly_best in enumerate(nearly_best)
                if is_nearly_best
            ]

        # If multiple candidates remain, choose one randomly
        return individuals[random.choice(candidates)]

    # 3. Pareto-based Tournament Selection for Initial Parent A Population
    def pareto_tournament_selection(population, k, tournsize=3):
        selected = []
        for _ in range(k):
            tournament_indices = np.random.choice(
                len(population), tournsize, replace=False
            )
            tournament = [population[i] for i in tournament_indices]

            # Define Pareto dominance based on fitness and complexity (length)
            def dominates(ind1, ind2):
                return (
                    np.mean(ind1.case_values) <= np.mean(ind2.case_values)
                    and len(ind1) < len(ind2)
                ) or (
                    np.mean(ind1.case_values) < np.mean(ind2.case_values)
                    and len(ind1) <= len(ind2)
                )

            # Find non-dominated individuals in the tournament
            non_dominated = []
            for ind1 in tournament:
                is_dominated = False
                for ind2 in tournament:
                    if ind1 is not ind2 and dominates(ind2, ind1):
                        is_dominated = True
                        break
                if not is_dominated:
                    non_dominated.append(ind1)

            # If multiple non-dominated individuals, choose one randomly. If none exist (very rare) pick a random individual from the tournament
            if non_dominated:
                selected.append(random.choice(non_dominated))
            else:
                selected.append(random.choice(tournament))

        return selected

    # parent_a = Select k//2 individuals for parent A, optionally considering residuals, diversity, and complexity.
    num_pareto = k // 4
    parent_a = pareto_tournament_selection(population, num_pareto)

    num_epsilon = k // 4
    for _ in range(num_epsilon):
        parent_a.append(epsilon_lexicase_selection(population))

    while len(parent_a) < k // 2:
        parent_a.append(random.choice(population))  # pad if needed

    # parent_b = Select k//2 individuals for parent B, considering complementarity in reducing error; diversity and complexity may also be considered.
    parent_b = []
    remaining_population_indices = list(range(pop_size))

    for _ in range(k // 2):
        if not remaining_population_indices:
            # if we run out of candidates, just randomly select from the population
            parent_b.append(population[np.random.choice(range(pop_size))])
            continue

        complementarity_scores = []
        for j in remaining_population_indices:
            # Measure improvement in error reduction if individual j is paired with a random individual from parent_a.
            individual_j = population[j]

            # choose a random parent A
            random_parent_a = random.choice(parent_a)

            # Spearman correlation of residuals to promote complementarity
            corr, _ = spearmanr(
                random_parent_a.y - random_parent_a.predicted_values,
                individual_j.y - individual_j.predicted_values,
            )

            # If correlation is NaN, set to a large value to avoid selecting this individual
            if np.isnan(corr):
                corr = 1.0

            complementarity_scores.append(corr)

        complementarity_scores = np.array(complementarity_scores)

        # Lower (more negative) correlation means better complementarity
        best_index_within_remaining = np.argmin(complementarity_scores)
        selected_index = remaining_population_indices[best_index_within_remaining]
        parent_b.append(population[selected_index])

        # Remove selected index to avoid duplicate selections
        remaining_population_indices.pop(best_index_within_remaining)

    # Interleave parent_a and parent_b to form crossover pairs
    selected_individuals = [
        individual for pair in zip(parent_a, parent_b) for individual in pair
    ]

    # if the length of selected_individuals is less than k, then we have to pad it with random individuals
    if len(selected_individuals) < k:
        num_to_pad = k - len(selected_individuals)
        for _ in range(num_to_pad):
            selected_individuals.append(population[np.random.choice(range(pop_size))])

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
