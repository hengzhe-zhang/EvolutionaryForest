import random

import numpy as np
from deap import tools

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast

from scipy.special import softmax


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
    Hybrid symbolic regression parent selector synthesizing strengths of Operators A & B:

    Key Features:
      - Adaptive nonlinear epsilon-lexicase filtering with early tightening + slight late relaxation.
      - Greedy coverage-based Parent A selection promoting diverse specialization.
      - Stochastic fallback tournament for Parent A fill robustness.
      - Parent B selection via annealed softmax sampling balancing residual weak spots,
        complementarity, residual diversity, and soft adaptive complexity penalty.
      - Strict duplicate avoidance for Parent B.
      - Complementarity and diversity explicitly included to promote crossover synergy.
      - Complexity penalty annealed and applied softly as tie-break and score component.
      - Clear stopping conditions to handle edge cases.
      - Vectorized numpy operations, minimal loops.

    Args:
        population (list): Individuals with attributes:
            - case_values (np.array): mse/errors per case (lower better)
            - residual (np.array): y - predicted_values residual vector
            - predicted_values (np.array)
            - height (int)
            - __len__() or number_of_nodes (int)
            - y (np.array): true target values, consistent across population
        k (int): even positive integer, number of total parents to select (paired)
        status (dict): expects "evolutionary_stage" float in [0,1]

    Returns:
        list: k selected individuals interleaved as [parent_a0, parent_b0, parent_a1, parent_b1, ...]
    """
    rng = np.random.default_rng()
    N = len(population)
    if N < 2 or k < 2:
        return []
    k = min(k, 2 * N)
    if k % 2 != 0:
        k -= 1
    if k == 0:
        return []

    evo_stage = float(status.get("evolutionary_stage", 0.0))
    evo_stage = np.clip(evo_stage, 0.0, 1.0)

    # --- Extract population data arrays ---
    fitness_matrix = np.stack(
        [np.asarray(ind.case_values) for ind in population]
    )  # (N,C)
    residual_matrix = np.stack(
        [np.asarray(ind.residual) for ind in population]
    )  # (N,C)
    y_true = population[0].y
    C = fitness_matrix.shape[1]

    nodes = np.array(
        [
            len(ind) if hasattr(ind, "__len__") else getattr(ind, "number_of_nodes", 0)
            for ind in population
        ],
        dtype=float,
    )
    height = np.array([getattr(ind, "height", 0) for ind in population], dtype=float)

    # Normalize complexities to [0,1]
    nodes_norm = (nodes - np.min(nodes)) / max(np.ptp(nodes), 1e-12)
    height_norm = (height - np.min(height)) / max(np.ptp(height), 1e-12)
    complexity = 0.5 * nodes_norm + 0.5 * height_norm  # combined complexity (0=best)

    # --- Adaptive epsilon schedule (based on Operator A nonlinear + stability) ---
    # Early tightening: ~1.0 -> 0.4 by evo=0.55; slight relaxation to ~0.6 by evo=1.0
    if evo_stage < 0.55:
        eps_factor = 1.0 - 1.6 * (evo_stage / 0.55)
    else:
        eps_factor = 0.4 + 0.2 * ((evo_stage - 0.55) / 0.45)
    eps_factor = np.clip(eps_factor, 0.3, 1.0)

    median_abs_residuals = np.median(np.abs(residual_matrix), axis=0)  # (C,)
    epsilon = median_abs_residuals * eps_factor  # adaptive epsilon per case

    # Identify specialization mask: True if individual's residual for case <= min + epsilon
    min_residuals = np.min(residual_matrix, axis=0)  # (C,)
    spec_mask = residual_matrix <= (min_residuals + epsilon)  # (N,C)
    spec_scores = spec_mask.sum(axis=1) / C  # fraction of cases specialized

    # --- Parent A Selection: Greedy coverage of specialized cases + soft complexity tie-break + fallback tournament ---

    # Candidates ordered descending by specialization fraction
    candidate_order = np.argsort(-spec_scores)
    selected_a_idx = []
    covered_cases = np.zeros(C, dtype=bool)

    # Soft complexity penalty weight diminishes as evolution progresses
    complexity_weight_a = 0.15 * (1.0 - evo_stage)  # tie-break weight

    # Greedy coverage selection loop with minimal iterations (early break)
    for idx in candidate_order:
        if len(selected_a_idx) >= (k // 2):
            break
        new_coverage = spec_mask[idx] & (~covered_cases)
        # Accept if covers new cases or if no parents selected yet
        if new_coverage.any() or len(selected_a_idx) == 0:
            selected_a_idx.append(idx)
            covered_cases |= spec_mask[idx]

    # Backup fill ignoring coverage but favor higher specialization - with soft complexity tie-break
    if len(selected_a_idx) < (k // 2):
        remaining = [idx for idx in candidate_order if idx not in selected_a_idx]
        # Calculate tie-break score: specialization_fraction - complexity_weight * complexity
        # Higher specialization + lower complexity preferred
        tie_scores = (
            spec_scores[remaining] - complexity_weight_a * complexity[remaining]
        )
        rem_order = np.argsort(-tie_scores)
        for i in rem_order:
            selected_a_idx.append(remaining[i])
            if len(selected_a_idx) >= (k // 2):
                break

    # Fallback stochastic tournament if still insufficient
    if len(selected_a_idx) < (k // 2):
        remaining = [i for i in range(N) if i not in selected_a_idx]
        if remaining:
            spec_norm = (spec_scores - spec_scores.min()) / max(
                np.ptp(spec_scores), 1e-12
            )
            tournament_size = int(np.interp(evo_stage, [0, 1], [5, 3]))
            while len(selected_a_idx) < (k // 2):
                participants = rng.choice(
                    remaining, min(tournament_size, len(remaining)), replace=False
                )
                winner = participants[np.argmax(spec_norm[participants])]
                selected_a_idx.append(winner)
                remaining.remove(winner)

    selected_a_idx = np.array(selected_a_idx[: (k // 2)])

    # --- Parent B Selection: Annealed softmax sampling on composite scoring ---
    # Components:
    #   - residual weak spots mean error (lower better)
    #   - complementarity norm (lower better)
    #   - residual diversity (higher better)
    #   - complexity penalty (lower complexity better)
    # Annealed weights modulated by evo_stage for balance.

    residuals_centered = residual_matrix - residual_matrix.mean(axis=1, keepdims=True)
    norm_residuals = np.linalg.norm(residuals_centered, axis=1) + 1e-12

    used_b = set()
    selected_b_idx = []

    # Annealing weights for components
    complexity_weight_b = (
        0.6 * (1.0 - evo_stage) + 0.15 * evo_stage
    )  # stronger early, lighter late
    diversity_weight = (
        0.25 * (1.0 - evo_stage) + 0.4 * evo_stage
    )  # diversity gains weight late
    complementarity_weight = (
        0.5 * (1.0 - evo_stage) + 0.35 * evo_stage
    )  # complementarity balanced
    residual_weight = (
        0.4 * (1.0 - evo_stage) + 0.6 * evo_stage
    )  # residual weak spots more important late

    # Temperature annealing for softmax: high early, low late for exploration->exploitation
    temperature = np.interp(evo_stage, [0, 1], [2.8, 0.5])

    for a_idx in selected_a_idx:
        a_residual = residual_matrix[a_idx]
        a_res_centered = residuals_centered[a_idx]
        a_norm = norm_residuals[a_idx]

        # Define weak spots: residuals above 75th percentile of a_idx residuals
        weak_threshold = np.percentile(a_residual, 75)
        weak_spots = a_residual > weak_threshold
        if not np.any(weak_spots):
            weak_spots = np.ones_like(weak_spots, dtype=bool)  # fallback all cases

        # Mean residuals on weak spots: lower better
        residual_scores = residual_matrix[:, weak_spots].mean(axis=1)

        # Complementarity (how well residuals offset each other): sum of abs(a_residual + candidate residual)
        combined_abs = np.abs(a_residual[None, :] + residual_matrix)
        combined_sum = combined_abs.sum(axis=1)
        combined_norm = (combined_sum - combined_sum.min()) / max(
            np.ptp(combined_sum), 1e-12
        )

        # Diversity: cosine distance (1 - cosine similarity)
        dotprod = residuals_centered @ a_res_centered
        cosine_sim = dotprod / (norm_residuals * a_norm)
        diversity_score = 1.0 - cosine_sim  # higher = more diverse

        # Composite raw score (lower better)
        raw_score = (
            residual_weight * residual_scores
            + complementarity_weight * combined_norm
            - diversity_weight * diversity_score
            + complexity_weight_b * complexity
        )

        # Exclude self and duplicates strictly
        raw_score[a_idx] = np.inf
        for used_idx in used_b:
            raw_score[used_idx] = np.inf

        valid_mask = np.isfinite(raw_score)
        if np.any(valid_mask):
            neg_scores = -raw_score[valid_mask]
            probs = softmax(neg_scores * temperature)
            # Stability check for probs
            if np.sum(probs) < 1e-14 or np.any(np.isnan(probs)):
                candidates = np.where(valid_mask)[0]
                chosen_rel = rng.choice(candidates)
                chosen_b = np.flatnonzero(valid_mask)[chosen_rel]
            else:
                candidate_indices = np.where(valid_mask)[0]
                chosen_b = rng.choice(candidate_indices, p=probs)
        else:
            # Fallback relax: allow duplicates but not self
            fallback_mask = np.ones(N, dtype=bool)
            fallback_mask[a_idx] = False
            candidates = np.where(fallback_mask)[0]
            chosen_b = rng.choice(candidates)

        selected_b_idx.append(chosen_b)
        used_b.add(chosen_b)

    selected_b_idx = np.array(selected_b_idx)

    # --- Form final selection interleaving Parent A and Parent B ---
    parent_a = [population[i] for i in selected_a_idx]
    parent_b = [population[i] for i in selected_b_idx]

    selected_individuals = [ind for pair in zip(parent_a, parent_b) for ind in pair]

    assert len(selected_individuals) == k, (
        f"Expected {k} selected but got {len(selected_individuals)}"
    )

    return selected_individuals


def novel_selection_plus(population, k=1, status={}):
    """
    Hybrid adaptive specialization + diversity coverage + complementarity + stage-aware complexity
    selection for symbolic regression genetic programming offspring pairing.

    Key design:
    - Parent A: lexicase-inspired specialization fraction (no complexity penalty),
      greedy coverage with max_iters guard for diverse specialized niches
    - Parent B: complementarity on parent's weak cases (top 25% residuals),
      complexity penalty scaled up with evolutionary stage (0.3→1.0)
    - Combined complexity: normalized avg of number_of_nodes and height
    - Avoid duplicates in parent_b to maintain diversity
    - Adaptive epsilon threshold using median absolute residual and evo stage
    - Interleaved output [parent_a0, parent_b0, parent_a1, parent_b1, ...]
    - Clear fallback handling and efficient numpy vectorization

    Args:
        population: list of individuals with attributes:
            - case_values: per-case MSE/errors (lower better) (array-like)
            - residual: per-case residuals (y - predicted) (array-like)
            - predicted_values
            - number_of_nodes or fallback len(individual)
            - height (int)
        k: even int number of individuals to select (paired: k//2 parent_a, k//2 parent_b)
        status: dict; must contain 'evolutionary_stage' key in [0,1]

    Returns:
        list of k selected individuals interleaved as [parent_a0, parent_b0, parent_a1, parent_b1, ...]
    """

    if len(population) == 0 or k == 0:
        return []

    pop_size = len(population)
    max_k = pop_size * 2 if pop_size > 0 else 0
    k = min(k, max_k)
    if k % 2 != 0:
        k -= 1
    if k == 0:
        return []

    evo_stage = float(status.get("evolutionary_stage", 0.0))
    evo_stage = np.clip(evo_stage, 0, 1)

    # Extract fitness (per case MSE) and residuals matrix (N, C)
    fitness_matrix = np.array([ind.case_values for ind in population])  # shape (N,C)
    residual_matrix = np.array([ind.residual for ind in population])  # shape (N,C)

    # Extract complexity metrics: number_of_nodes and height
    complexities = np.array(
        [
            getattr(ind, "number_of_nodes", len(ind) if hasattr(ind, "__len__") else 0)
            for ind in population
        ],
        dtype=float,
    )
    heights = np.array([getattr(ind, "height", 0) for ind in population], dtype=float)

    # Normalize complexity metrics with epsilon to avoid div by zero
    comp_range = max(np.ptp(complexities), 1e-8)
    height_range = max(np.ptp(heights), 1e-8)

    comp_norm = (complexities - complexities.min()) / comp_range
    height_norm = (heights - heights.min()) / height_range

    complexity = 0.5 * comp_norm + 0.5 * height_norm  # combined complexity [0,1]

    num_cases = fitness_matrix.shape[1]

    # Adaptive epsilon threshold for lexicase specialization mask
    median_abs_residuals = np.median(np.abs(residual_matrix), axis=0)  # shape (C,)
    epsilon = median_abs_residuals * (0.2 + 0.8 * evo_stage)

    min_residuals = np.min(residual_matrix, axis=0)  # shape (C,)

    # Parent A - specialization mask (boolean pass per case)
    pass_mask = residual_matrix <= (min_residuals + epsilon)
    specialization_score = pass_mask.sum(axis=1) / num_cases  # fraction [0,1]

    # Parent A selection: greedy coverage of specialized cases, no complexity penalty applied here
    sorted_candidates = np.argsort(
        -specialization_score
    )  # descending order specialization

    selected_a_idx = []
    covered_cases = np.zeros(num_cases, dtype=bool)
    max_iters = 1000
    iters = 0

    # Greedy coverage loop: select individuals adding new specialty coverage
    while len(selected_a_idx) < (k // 2) and iters < max_iters:
        for idx in sorted_candidates:
            if len(selected_a_idx) >= (k // 2):
                break
            new_cases = pass_mask[idx] & (~covered_cases)
            if new_cases.any() or len(selected_a_idx) == 0:
                selected_a_idx.append(idx)
                covered_cases |= pass_mask[idx]
        iters += 1
        if iters >= max_iters:
            break

    # Backup fill if not enough selected ignoring coverage now
    if len(selected_a_idx) < (k // 2):
        existing = set(selected_a_idx)
        for idx in sorted_candidates:
            if idx not in existing:
                selected_a_idx.append(idx)
                if len(selected_a_idx) >= (k // 2):
                    break

    selected_a_idx = selected_a_idx[: k // 2]

    # Prepare Parent B selection
    # Complexity penalty scaled progressively higher late evo phase: from 0.3 to 1.0
    comp_penalty_scale_b = 0.3 + 0.7 * evo_stage
    complexity_penalty_b = complexity * comp_penalty_scale_b

    selected_b_idx = []
    used_b = set()

    for a_idx in selected_a_idx:
        a_residual = residual_matrix[a_idx]
        threshold_75 = np.percentile(a_residual, 75)
        weak_mask = a_residual > threshold_75

        # If no weak cases, consider all cases to avoid empty mask for mean
        if not weak_mask.any():
            weak_mask = np.ones_like(a_residual, dtype=bool)

        # Score candidates by mean residual over parent's weak cases + complexity penalty
        scores_b = residual_matrix[:, weak_mask].mean(axis=1) + complexity_penalty_b

        # Avoid self and duplicates
        scores_b[a_idx] = np.inf
        for ub in used_b:
            scores_b[ub] = np.inf

        best_b_idx = np.argmin(scores_b)

        # Fallback if all inf (no suitable candidate)
        if scores_b[best_b_idx] == np.inf:
            candidates = [i for i in range(pop_size) if i != a_idx and i not in used_b]
            if not candidates:
                candidates = [i for i in range(pop_size) if i != a_idx]
            best_b_idx = np.random.choice(candidates)

        selected_b_idx.append(best_b_idx)
        used_b.add(best_b_idx)

    # Build parent lists
    parent_a = [population[i] for i in selected_a_idx]
    parent_b = [population[i] for i in selected_b_idx]

    # Interleave for paired crossover
    selected_individuals = [ind for pair in zip(parent_a, parent_b) for ind in pair]
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
