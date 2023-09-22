import numpy as np

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast


def doubleLexicase(pop, k, lexicase_round=10, size_selection='Roulette'):
    chosen = []
    for _ in range(k):
        candidates = selAutomaticEpsilonLexicaseFast(pop, lexicase_round)
        if hasattr(candidates[0], 'fitness') and len(candidates[0].fitness.wvalues) > 1:
            # For multi-object optimization, this might be a good way
            size_arr = [x.fitness.wvalues[1] for x in candidates]
            # change maximize to minimize
            size_arr = np.array([-x for x in size_arr])
        else:
            size_arr = np.array([len(x) for x in candidates])
        if size_selection == 'Roulette':
            size_arr = np.max(size_arr) + np.min(size_arr) - size_arr
            if size_arr.sum() <= 0:
                index = np.random.choice([i for i in range(0, len(size_arr))])
            else:
                index = np.random.choice([i for i in range(0, len(size_arr))], p=size_arr / size_arr.sum())
        elif size_selection == 'Min':
            index = np.argmin(size_arr)
        else:
            raise Exception('Unknown Size Selection Operator!')
        chosen.append(candidates[index])
    return chosen
