import random
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def selBoltzmann(
    individuals,
    k,
    model: "EvolutionaryForestRegressor",
    N=10000,
    T0=0.1,
    fit_attr="fitness",
):
    """
    Boltzmann selection with temperature scheduling.

    Args:
        individuals: list of individuals (must have .fitness.wvalues[0])
        k: number of individuals to select
        u: number of individuals evaluated so far (for temperature scheduling)
        N: normalization constant for scheduling (default: 10,000)
        T0: initial temperature (default: 0.1)
        fit_attr: fitness attribute to sort/select on (default: 'fitness')
    """
    u = model.current_gen
    # Calculate dynamic temperature
    tau = T0 * (1 - (u % N) / N)

    # Avoid divide-by-zero issues
    if tau <= 1e-8:
        tau = 1e-8

    # Compute Boltzmann weights
    scores = [getattr(ind, fit_attr).wvalues[0] for ind in individuals]
    exp_scores = [math.exp(score / tau) for score in scores]
    sum_exp = sum(exp_scores)

    # Normalize weights to get probabilities
    probabilities = [w / sum_exp for w in exp_scores]

    # Sample k individuals with weighted probabilities
    chosen = random.choices(individuals, weights=probabilities, k=k)
    return chosen
