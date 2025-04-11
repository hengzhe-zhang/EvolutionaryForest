import random

import numpy as np
from deap import tools

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
    Novel selection operator for genetic programming, improved from Operator A and B,
    incorporating diverse and specialized selection, crossover-aware pairing,
    stage-specific pressure, interpretability preference, and efficiency.

    This operator selects parent A using a stage-aware tournament selection that considers
    fitness, interpretability, and specialization. Parent B is then selected to be
    complementary to Parent A based on residual anti-correlation.

    Args:
        population (list): List of individual objects.
        k (int): Number of individuals to select.
        status (dict): Dictionary containing evolutionary status, e.g.,
                       {"evolutionary_stage": [0, 1]}.

    Returns:
        list: List of selected individuals.
    """
    selected = []
    pop_size = len(population)
    if pop_size == 0:
        return selected

    evolutionary_stage = status.get("evolutionary_stage", 0)  # Default to early stage

    # --- 1. Stage-Specific Weighting ---
    # Adjust weights to emphasize different objectives across evolutionary stages.
    fitness_weight = (
        0.6 + evolutionary_stage * 0.4
    )  # Fitness becomes more important later
    interpretability_weight = 0.3 * (
        1.0 - evolutionary_stage
    )  # Interpretability is crucial early
    specialization_weight = 0.1 * (
        1.0 - evolutionary_stage
    )  # Specialization is valuable early

    parent_a_list = []
    num_parent_a = k // 2

    # --- 2. Parent A Selection: Stage-Aware Tournament with Multi-Objective Score ---
    for _ in range(num_parent_a):
        tournament_pool = np.random.choice(
            population, size=min(5, pop_size), replace=False
        )
        best_parent_a = None
        best_tournament_score = -float("inf")

        for individual_a in tournament_pool:
            # --- Define Objectives ---
            fitness = -np.mean(
                individual_a.case_values
            )  # Lower case_values (error) is better
            interpretability = (
                len(individual_a) + individual_a.height
            )  # Smaller trees are preferred
            specialization = (
                -np.var(individual_a.case_values)
                if len(individual_a.case_values) > 1
                else 0
            )  # Lower variance means more specialized

            # --- Calculate Stage-Aware Tournament Score ---
            tournament_score = (
                (fitness_weight * fitness)
                + (interpretability_weight * (-interpretability))
                + (specialization_weight * specialization)
            )

            # --- Tournament Winner Selection ---
            if tournament_score > best_tournament_score:
                best_tournament_score = tournament_score
                best_parent_a = individual_a

        # --- Handle Cases where Tournament Might Fail (Edge Cases) ---
        if best_parent_a is not None:
            parent_a_list.append(best_parent_a)
        elif tournament_pool.size > 0:
            parent_a_list.append(
                tournament_pool[0]
            )  # Fallback: if no best found, take first from pool
        elif population:
            parent_a_list.append(
                np.random.choice(population)
            )  # Fallback: if pool empty, random from population

    # --- 3. Parent B Selection: Residual-Based Complementary Pairing ---
    parent_b_list = []
    for mother in parent_a_list:
        best_father = None
        best_complementarity_score = -float("inf")
        mother_residual = mother.y - mother.predicted_values  # Error vector of parent A

        for father_candidate in population:
            if father_candidate is mother:  # Avoid self-pairing
                continue

            father_residual = (
                father_candidate.y - father_candidate.predicted_values
            )  # Error vector of candidate parent B

            # --- Complementarity Metric: Anti-correlation of Residuals (Cosine Similarity) ---
            mother_residual_norm = mother_residual / (
                np.linalg.norm(mother_residual) + 1e-8
            )  # Normalize residuals to unit vectors
            father_residual_norm = father_residual / (
                np.linalg.norm(father_residual) + 1e-8
            )
            complementarity_score = -np.dot(
                mother_residual_norm, father_residual_norm
            )  # Maximize negative cosine similarity (anti-correlation)

            # --- Select Best Complementary Father ---
            if complementarity_score > best_complementarity_score:
                best_complementarity_score = complementarity_score
                best_father = father_candidate
            elif (
                complementarity_score == best_complementarity_score
                and best_father is not None
            ):
                # Tie-breaker: Favor simpler fathers for interpretability
                if (len(father_candidate) + father_candidate.height) < (
                    len(best_father) + best_father.height
                ):
                    best_father = father_candidate
            elif best_father is None:
                best_father = (
                    father_candidate  # If no best father yet, assign current as best.
                )

        # --- Handle Cases where Complementary Father Might Not Be Found ---
        if best_father is not None:
            parent_b_list.append(best_father)
        else:
            parent_b_list.append(
                mother
            )  # Fallback: Duplicate mother if no father found (rare, but robust)

    # --- 4. Combine Parent A and Parent B Lists to Create Selected Individuals ---
    # Interleave parent A and parent B to form pairs and then flatten the list.
    selected_individuals = [
        val for pair in zip(parent_a_list, parent_b_list) for val in pair
    ][:k]
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
