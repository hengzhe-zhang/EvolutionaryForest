import numpy as np
from scipy.special import softmax

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast


def get_individual_size(x):
    """Return a size metric suitable for minimization."""
    if (
        hasattr(x, "fitness")
        and hasattr(x.fitness, "wvalues")
        and len(x.fitness.wvalues) > 1
    ):
        # Negate because DEAP maximizes by default
        return -x.fitness.wvalues[1]
    return sum(len(tree) for tree in x.gene)


def doubleLexicase(pop, k, lexicase_round=10, size_selection="Roulette"):
    chosen = []
    objective_std = np.std([get_individual_size(x) for x in pop])

    for _ in range(k):
        candidates = selAutomaticEpsilonLexicaseFast(pop, lexicase_round)
        size_arr = np.array([get_individual_size(x) for x in candidates])  # minimize

        if size_selection == "Softmax":
            # because it's a minimization problem, we need to convert back
            if objective_std > 0:
                size_arr /= objective_std
            index = np.random.choice(range(len(size_arr)), p=softmax(-1 * size_arr))
        elif size_selection == "Roulette":
            size_arr = np.max(size_arr) + np.min(size_arr) - size_arr
            if size_arr.sum() <= 0:
                index = np.random.choice(range(len(size_arr)))
            else:
                index = np.random.choice(
                    range(len(size_arr)), p=size_arr / size_arr.sum()
                )
        elif size_selection == "Min":
            index = np.argmin(size_arr)
        else:
            raise Exception("Unknown Size Selection Operator!")
        chosen.append(candidates[index])
    return chosen
