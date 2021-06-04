import copy
import random

import numpy as np
from deap.tools import selRandom


def batch_tournament_selection(individuals, k, tournsize, batch_size, fit_attr="fitness"):
    fitness_cases_num = len(individuals[0].case_values)
    idx_cases_batch = np.arange(0, fitness_cases_num)
    np.random.shuffle(idx_cases_batch)
    _batches = np.array_split(idx_cases_batch, max(fitness_cases_num // batch_size, 1))
    batch_ids = np.arange(0, len(_batches))
    assert len(_batches[0]) >= batch_size or fitness_cases_num < batch_size

    chosen = []
    while len(chosen) < k:
        batches: list = copy.deepcopy(batch_ids.tolist())
        while len(batches) > 0 and len(chosen) < k:
            idx_candidates = selRandom(individuals, tournsize)
            cand_fitness_for_this_batch = []
            for idx in idx_candidates:
                cand_fitness_for_this_batch.append(np.mean(idx.case_values[_batches[batches[0]]]))
            idx_winner = np.argmin(cand_fitness_for_this_batch)
            winner = idx_candidates[idx_winner]
            chosen.append(winner)
            batches.pop(0)
    return chosen


def selAutomaticEpsilonLexicase(individuals, k):
    selected_individuals = []

    for i in range(k):
        candidates = individuals
        cases = list(range(len(individuals[0].case_values)))
        random.shuffle(cases)
        fit_weights = individuals[0].fitness.weights[0]

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = [x.case_values[cases[0]] for x in candidates]
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
            if fit_weights > 0:
                best_val_for_case = np.max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list([x for x in candidates if x.case_values[cases[0]] >= min_val_to_survive])
            else:
                best_val_for_case = np.min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list([x for x in candidates if x.case_values[cases[0]] <= max_val_to_survive])
            cases.pop(0)
        # print('used cases', len(individuals[0].case_values) - len(cases))
        selected_individuals.append(random.choice(candidates))
    return selected_individuals


def selAutomaticEpsilonLexicaseK(individuals, k):
    candidates = individuals
    cases = list(range(len(individuals[0].case_values)))
    random.shuffle(cases)
    fit_weights = individuals[0].fitness.weights[0]

    while len(cases) > 0 and len(candidates) > k:
        errors_for_this_case = [x.case_values[cases[0]] for x in candidates]
        median_val = np.median(errors_for_this_case)
        median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
        if fit_weights > 0:
            best_val_for_case = np.max(errors_for_this_case)
            min_val_to_survive = best_val_for_case - median_absolute_deviation
            c = list([x for x in candidates if x.case_values[cases[0]] >= min_val_to_survive])
            if len(c) < k:
                break
            candidates = c
        else:
            best_val_for_case = np.min(errors_for_this_case)
            max_val_to_survive = best_val_for_case + median_absolute_deviation
            c = list([x for x in candidates if x.case_values[cases[0]] <= max_val_to_survive])
            if len(c) < k:
                break
            candidates = c
        cases.pop(0)
    return random.sample(candidates, k)


def selTournamentPlus(individuals, k, tournsize):
    """
    Select individuals based on the sum of case values
    :param individuals: population
    :param k: number of offspring
    :param tournsize: tournament size
    :return:
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(min(aspirants, key=lambda x: np.sum(x.case_values)))
    return chosen
