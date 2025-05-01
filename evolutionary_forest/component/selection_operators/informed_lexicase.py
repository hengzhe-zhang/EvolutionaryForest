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

    for _ in range(k // 2):
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
    stage = np.clip(status.get("evolutionary_stage", 0), 0, 1)
    n = len(population)
    k = min(k, n if n % 2 == 0 else n - 1)
    half_k = k // 2

    mses = np.array([ind.case_values.mean() for ind in population])
    nodes = np.array([len(ind) for ind in population])
    heights = np.array([ind.height for ind in population])
    complexity = nodes + heights
    preds = np.array([ind.predicted_values for ind in population])
    resid = np.array([ind.y - ind.predicted_values for ind in population])
    n_inst = preds.shape[1]

    mse_n = (mses - mses.min()) / (mses.ptp() + 1e-12)
    comp_n = (complexity - complexity.min()) / (complexity.ptp() + 1e-12)
    interp = 1 / (1 + complexity)

    w_fit = 0.7 * stage + 0.3 * (1 - stage)
    w_interp = 0.7 * (1 - stage) + 0.3 * stage
    w_div = 0.5 * (1 - stage)

    rng = np.random.default_rng(42 + int(stage * 1e5))
    size_sub_a = max(5, n_inst // 7)
    size_sub_b = max(5, n_inst // 6)
    idx_sub_a = rng.choice(n_inst, size_sub_a, replace=False)
    idx_sub_b = rng.choice(n_inst, size_sub_b, replace=False)

    preds_a = preds[:, idx_sub_a]
    preds_b = preds[:, idx_sub_b]

    # Diversity for parent A on subset A
    diff_a = preds_a[:, None, :] - preds_a[None, :, :]
    diversity_a = np.sqrt(np.mean(diff_a**2, axis=2)).mean(axis=1)

    # Composite score for parent A balancing fit, interpretability, diversity
    score_a = w_fit * (1 - mse_n) + w_interp * interp + w_div * diversity_a
    idx_a = np.argpartition(-score_a, half_k)[:half_k]
    parent_a = [population[i] for i in idx_a]

    max_comp = max(complexity.max(), 1)
    comp_penalty = 0.5 * (complexity[:, None] + complexity) / (2 * max_comp)

    parent_b = []
    for i_a in idx_a:
        res_a = resid[i_a, idx_sub_b]
        div_i = np.sqrt(np.mean((preds_b[i_a] - preds_b) ** 2, axis=1))
        mask = np.arange(n) != i_a
        combined_var = np.var(res_a + resid[:, idx_sub_b], axis=1)
        combined_var[~mask] = np.inf
        scores = combined_var + comp_penalty[i_a] - 0.3 * div_i
        b_idx = np.argmin(scores)
        parent_b.append(population[b_idx])

    selected_individuals = [ind for pair in zip(parent_a, parent_b) for ind in pair]
    return selected_individuals[:k]


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
