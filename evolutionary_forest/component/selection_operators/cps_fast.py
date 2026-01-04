import numba
import numpy as np
import random

from evolutionary_forest.component.selection import (
    selAutomaticEpsilonLexicaseNumba,
    selAutomaticEpsilonLexicaseFast,
)
from deap.tools import selTournament


# Numba-optimized function for selection of the best father
@numba.jit(nopython=True)
def select_best_father(
    mother_pred, father_preds, target, fit_weights, case_values, metric="MSE"
):
    best_father = -1
    best_fitness = -np.inf

    # Number of individuals in population
    n = father_preds.shape[0]

    # Iterate over fathers to calculate fitness and select best
    for i in range(n):
        # Skip if father is the same as mother
        if np.array_equal(father_preds[i], mother_pred):
            continue

        # Compute the average prediction
        avg_pred = (mother_pred + father_preds[i]) / 2

        # Calculate fitness based on selected metric
        if metric == "MSE":
            fit = -np.mean((avg_pred - target) ** 2)
        elif metric == "Pearson":
            if np.std(avg_pred) == 0 or np.std(target) == 0:
                fit = -1  # Avoid division by zero, treat as worst correlation
            else:
                fit = np.dot(avg_pred - np.mean(avg_pred), target - np.mean(target)) / (
                    np.linalg.norm(avg_pred - np.mean(avg_pred))
                    * np.linalg.norm(target - np.mean(target))
                )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        # Select best father
        if fit > best_fitness:
            best_father = i
            best_fitness = fit
        elif fit == best_fitness:
            if np.sum(case_values[i]) < np.sum(case_values[best_father]):
                best_father = i

    return best_father


# Numba-optimized function for the entire loop logic
@numba.jit(nopython=True)
def run_selection_loop(
    case_values, predicted_values, target, fit_weights, metric="MSE", k=1
):
    selected_indices = []

    for _ in range(k):
        # Select the first parent using epsilon-lexicase selection
        index, avg_cases = selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, 1)
        mother_idx = index[0]
        mother_pred = predicted_values[mother_idx]

        # Select the best father using Numba optimized function
        best_father_idx = select_best_father(
            mother_pred, predicted_values, target, fit_weights, case_values, metric
        )

        # Record the indices of selected mother and father
        selected_indices.extend([mother_idx, best_father_idx])

    return selected_indices


# Wrapper function for selecting parents with Numba
def select_cpsr_regression_fast(population, k, target, metric="MSE"):
    """
    CPS selection (residual version) for symbolic regression with vectorized NumPy operations.

    Assumes:
    - individual.case_values is a NumPy array of squared errors per case
    """
    # Convert population attributes to NumPy arrays for Numba processing
    predicted_values = np.array([ind.predicted_values for ind in population])
    case_values = np.array([ind.case_values for ind in population])
    fit_weights = population[0].fitness.weights[0]

    # Call the Numba-optimized selection loop
    selected_indices = run_selection_loop(
        case_values, predicted_values, target, fit_weights, metric, k
    )

    # Use the indices to retrieve the corresponding individuals from the population
    selected = [population[i] for i in selected_indices]

    return selected


def select_lexicase_random_second(population, k):
    """
    Variant 1: Lexicase selection for first parent, random selection for second parent.
    This is an ablation study variant to compare with P-CGS.

    Args:
        population: List of individuals
        k: Number of parent pairs to select (returns 2*k individuals)
        target: Target values for regression
        metric: Not used in this variant, kept for API consistency
    """
    selected = []

    for _ in range(k):
        # Select first parent using lexicase selection
        first_parent = selAutomaticEpsilonLexicaseFast(population, 1)[0]

        # Select second parent randomly (excluding the first parent)
        remaining_population = [ind for ind in population if ind is not first_parent]
        second_parent = random.choice(remaining_population)

        selected.extend([first_parent, second_parent])

    return selected


def select_tournament_cpsr(population, k, target, metric="Pearson", tournsize=7):
    """
    Variant 2: Tournament selection for first parent, P-CGS complementarity for second parent.
    This is an ablation study variant to compare with P-CGS.

    Args:
        population: List of individuals
        k: Number of parent pairs to select (returns 2*k individuals)
        target: Target values for regression
        metric: Complementarity metric ("Pearson" or "MSE")
        tournsize: Tournament size for first parent selection
    """
    selected = []
    predicted_values = np.array([ind.predicted_values for ind in population])
    case_values = np.array([ind.case_values for ind in population])
    fit_weights = population[0].fitness.weights[0]

    for _ in range(k):
        # Select first parent using tournament selection
        first_parents = selTournament(population, 1, tournsize=tournsize)
        first_parent = first_parents[0]
        first_parent_idx = population.index(first_parent)
        first_parent_pred = predicted_values[first_parent_idx]

        # Select second parent using complementarity (same as P-CGS)
        best_father_idx = select_best_father(
            first_parent_pred,
            predicted_values,
            target,
            fit_weights,
            case_values,
            metric,
        )
        second_parent = population[best_father_idx]

        selected.extend([first_parent, second_parent])

    return selected
