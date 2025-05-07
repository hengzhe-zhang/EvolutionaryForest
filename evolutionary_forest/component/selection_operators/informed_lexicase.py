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
    if n == 0 or k == 0:
        return []
    k = min(k, n - (k % 2))
    half_k = k // 2
    n_cases = population[0].y.shape[0]

    mse = np.array([np.mean(ind.case_values) for ind in population])
    nodes = np.array([len(ind) for ind in population])
    height = np.array([ind.height for ind in population])
    complexity = (nodes / max(nodes.max(), 1) + height / max(height.max(), 1)) / 2
    residuals = np.array([ind.y - ind.predicted_values for ind in population])

    def norm(x):
        r = x.ptp()
        return (x - x.min()) / (r + 1e-9)

    mse_n = norm(mse)
    diversity = np.array([np.std(res) for res in residuals])
    diversity_n = norm(diversity)

    fit_w = 0.4 + 0.6 * stage
    comp_w = 1 - fit_w

    subsets = np.array_split(np.random.permutation(n_cases), half_k)
    parent_a, used = [], set()
    for subset in subsets:
        subset_mse = np.array(
            [
                np.mean((ind.y[subset] - ind.predicted_values[subset]) ** 2)
                for ind in population
            ]
        )
        spec_score = norm(-subset_mse)
        score = fit_w * (1 - mse_n) + comp_w * (
            0.4 * (1 - complexity) + 0.35 * spec_score + 0.25 * diversity_n
        )
        sorted_idx = np.argsort(-score)
        for idx in sorted_idx:
            if idx not in used:
                parent_a.append(population[idx])
                used.add(idx)
                break
        else:
            parent_a.append(population[sorted_idx[0]])
            used.add(sorted_idx[0])

    parent_b = []
    diversity_max = max(diversity.max(), 1)
    nodes_max, height_max = max(nodes.max(), 1), max(height.max(), 1)
    complexity_raw = (nodes / nodes_max + height / height_max) / 2
    for ind_a in parent_a:
        res_a = ind_a.y - ind_a.predicted_values
        std_a = np.std(res_a)
        std_a = std_a if std_a > 1e-10 else 1
        res_b_all = residuals
        std_b_all = np.std(res_b_all, axis=1)
        std_b_all[std_b_all < 1e-10] = 1
        corr_num = (
            (res_b_all - res_b_all.mean(axis=1, keepdims=True))
            * (res_a - np.mean(res_a))
        ).mean(axis=1)
        corr = corr_num / (std_b_all * std_a)
        corr[np.isnan(corr)] = 0
        idx_a = np.where(np.array(population) == ind_a)[0][0]
        corr[idx_a] = np.inf
        fit_score = mse
        diversity_scaled = diversity / diversity_max
        combined_score = (
            np.abs(corr) * (1 - stage)
            + comp_w * (0.4 * complexity_raw + 0.25 * diversity_scaled)
            + fit_score * stage
        )
        combined_score[idx_a] = np.inf
        best_idx = np.argmin(combined_score)
        parent_b.append(population[best_idx])

    return [ind for pair in zip(parent_a, parent_b) for ind in pair]


def novel_selection_plus(population, k=100, status={}):
    stage = np.clip(status.get("evolutionary_stage", 0), 0, 1)
    n = len(population)
    k = min(k, (n // 2) * 2)
    m = len(population[0].case_values)

    errors = np.array([np.mean(ind.case_values) for ind in population])
    nodes = np.array([len(ind) for ind in population])
    heights = np.array([ind.height for ind in population])
    complexity = nodes + heights + 1e-8
    comp_norm = complexity / complexity.max()

    # Stage-varying weights for fitness, specialization, and complexity
    w_fit = 1 - 0.7 * stage
    w_spec = 0.3 + 0.7 * stage
    w_comp = 0.1 + 0.85 * stage

    # Construct diverse, mostly disjoint subsets with fallback to full range
    subset_size = max(3, m // 8)
    num_subsets = k // 2
    used = set()
    subsets = []
    for _ in range(num_subsets):
        avail = [i for i in range(m) if i not in used]
        if len(avail) < subset_size:
            avail = list(range(m))
        sub = np.random.choice(avail, subset_size, replace=False)
        used.update(sub)
        subsets.append(sub)
    subsets = np.array(subsets)

    # Compute subset MSE matrix and relative performance
    mse_sub = np.array(
        [[np.mean(ind.case_values[s]) for ind in population] for s in subsets]
    )
    rel_perf = errors / (mse_sub + 1e-8)

    # Score: weighted sum of inverted subset error, relative perf., and inverse complexity
    scores = w_fit / (mse_sub + 1e-8) + w_spec * rel_perf + w_comp / complexity
    parent_a_idx = np.argmax(scores, axis=1)
    parent_a = [population[i] for i in parent_a_idx]

    residuals = np.array([ind.y - ind.predicted_values for ind in population])
    norms = np.linalg.norm(residuals, axis=1) + 1e-10

    parent_b_idx = []
    for i in parent_a_idx:
        res_a = residuals[i]
        norm_a = norms[i]
        corrs = residuals @ res_a / (norms * norm_a)  # cosine similarity
        # Promote complementarity: minimize absolute correlation plus complexity penalty
        combined = np.abs(corrs) + w_comp * comp_norm
        combined[i] = np.inf
        parent_b_idx.append(np.argmin(combined))
    parent_b = [population[i] for i in parent_b_idx]

    # Interleave parents into pairs
    return [ind for pair in zip(parent_a, parent_b) for ind in pair][:k]


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
