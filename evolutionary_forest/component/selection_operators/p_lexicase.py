import numpy as np


def n_plexicase_selection(population, k, temperature=1.0):
    """
    n-Plexicase selection using MAD-based epsilon and softmax sampling.

    Args:
        population: list of individuals with .case_values attribute (shape: num_cases).
        k: int, number of individuals to select.
        temperature: float, softmax temperature to control selection pressure.

    Returns:
        List of selected individuals.
    """
    if len(population) == 0:
        return []

    # Step 1: Build fitness matrix (shape: N individuals x C cases)
    fitness_matrix = np.array([ind.case_values for ind in population])
    n_cand, n_cases = fitness_matrix.shape

    # Step 2: Compute MAD-based epsilon (for minimization)
    medians = np.median(fitness_matrix, axis=0)
    epsilon = np.median(np.abs(fitness_matrix - medians), axis=0)

    # Step 3: Use modified plexicase to find non-dominated individuals and their probabilities
    nondom_set, probs = plexicase(
        fitness_matrix, epsilon, greater_is_better=False, alpha=temperature
    )

    # Step 4: Sample from non-dominated individuals
    selected_indices = np.random.choice(nondom_set, size=k, p=probs, replace=False)
    return [population[i] for i in selected_indices]


def plexicase(
    fitness_matrix: np.array,
    epsilon: np.array,
    greater_is_better: bool,
    alpha: float = 1.0,
):
    """Epsilon-plexicase selection.

    Args:
        fitness_matrix (np.array): Fitness matrix of shape (n_cand, n_cases).
        epsilon (np.array): Epsilon vector of shape (n_cases,).
        greater_is_better (bool): Whether greater fitness is better.
        alpha (float, optional): Alpha parameter. Defaults to 1..

    Returns:
        Tuple of (indices of non-dominated candidates, probability vector of non-dominated candidates).
    """
    n_cand, n_cases = fitness_matrix.shape

    if greater_is_better:
        fitness_threshold = np.max(fitness_matrix, axis=0) - epsilon
        fit_is_best = fitness_matrix >= fitness_threshold
    else:
        fitness_threshold = np.min(fitness_matrix, axis=0) + epsilon
        fit_is_best = fitness_matrix <= fitness_threshold
    n_best = np.sum(fit_is_best, axis=1)

    unchecked = np.argsort(n_best)
    dom_set = []
    total_comparisons = 0

    while len(unchecked) > 0:
        idx = unchecked[0]

        cur_n_best = n_best[idx]
        to_compare = np.arange(n_cand)[n_best <= cur_n_best]
        # remove self
        to_compare = to_compare[to_compare != idx]
        # remove already dominated cand
        to_compare = to_compare[np.logical_not(np.isin(to_compare, dom_set))]

        if len(to_compare) > 0:
            total_comparisons += len(to_compare)

            cur_fit = fitness_matrix[idx]
            to_compare_fit = fitness_matrix[to_compare]

            cur_fit_is_best = fit_is_best[idx]
            to_compare_fit_is_best = fit_is_best[to_compare]

            if greater_is_better:
                # A - current; B - to compare
                # tie on some best and B is better on something
                cond1 = (
                    np.sum(cur_fit_is_best * to_compare_fit_is_best, axis=1)
                    * np.sum((to_compare_fit > cur_fit + epsilon), axis=1)
                ) > 0
            else:
                # tie on some best and B is better on something
                cond1 = (
                    np.sum(cur_fit_is_best * to_compare_fit_is_best, axis=1)
                    * np.sum((to_compare_fit < cur_fit - epsilon), axis=1)
                ) > 0

            # B is better on something with best error
            cond2 = np.sum(to_compare_fit_is_best * (1 - cur_fit_is_best), axis=1) > 0

            dom_set += list(to_compare[np.logical_not(np.logical_or(cond1, cond2))])

        unchecked = unchecked[unchecked != idx]  # remove self
        unchecked = unchecked[np.logical_not(np.isin(unchecked, dom_set))]

    nondom_set = np.arange(n_cand)
    nondom_set = nondom_set[np.logical_not(np.isin(nondom_set, dom_set))]
    n_best_nondom = n_best[nondom_set]

    if greater_is_better:
        best_each_case = np.array(
            [
                fitness_matrix[nondom_set, i] >= fitness_threshold[i]
                for i in range(n_cases)
            ]
        ).astype(float)  # (n_cases, nondom set)
    else:
        best_each_case = np.array(
            [
                fitness_matrix[nondom_set, i] <= fitness_threshold[i]
                for i in range(n_cases)
            ]
        ).astype(float)  # (n_cases, nondom set)

    p_each_case = best_each_case * n_best_nondom
    p_each_case = p_each_case / np.sum(p_each_case, axis=1, keepdims=True)
    p = np.sum(p_each_case, axis=0)

    # normalize
    p = p / np.sum(p)

    # manipulate
    p = np.power(p, alpha)
    p = p / np.sum(p)

    return nondom_set, p
