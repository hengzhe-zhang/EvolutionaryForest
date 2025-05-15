import numpy as np
from numba import njit


@njit(cache=True)
def find_min_variance_threshold(errors):
    best_tau = errors[0]
    best_score = np.inf

    sorted_errors = np.sort(errors)
    n = len(errors)

    for i in range(1, n):
        tau = (sorted_errors[i - 1] + sorted_errors[i]) / 2.0

        left = errors[errors < tau]
        right = errors[errors >= tau]

        if len(left) == 0 or len(right) == 0:
            continue

        left_var = np.var(left) if len(left) > 1 else 0.0
        right_var = np.var(right) if len(right) > 1 else 0.0

        score = (left_var * len(left) + right_var * len(right)) / n
        if score < best_score:
            best_score = score
            best_tau = tau

    return best_tau


@njit(cache=True)
def selMinimumVarianceLexicaseNumba(case_values, fit_weights, k):
    selected_individuals = []

    for i in range(k):
        candidates = list(range(len(case_values)))
        cases = np.arange(len(case_values[0]))
        np.random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = np.array(
                [case_values[x][cases[0]] for x in candidates]
            )
            tau_star = find_min_variance_threshold(errors_for_this_case)

            if fit_weights > 0:
                candidates = [
                    x for x in candidates if case_values[x][cases[0]] >= tau_star
                ]
            else:
                candidates = [
                    x for x in candidates if case_values[x][cases[0]] <= tau_star
                ]

            cases = np.delete(cases, 0)

        selected_individuals.append(np.random.choice(np.array(candidates)))

    return selected_individuals


def selMinimumVarianceLexicaseFast(individuals, k):
    fit_weights = individuals[0].fitness.weights[0]
    case_values = np.array([ind.case_values for ind in individuals])
    selected_indices = selMinimumVarianceLexicaseNumba(case_values, fit_weights, k)
    return [individuals[i] for i in selected_indices]
