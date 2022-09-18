import copy
import random
from types import SimpleNamespace

import numpy as np
import seaborn as sns
from deap.tools import selRandom, selTournament
from matplotlib import pyplot as plt
from numba import njit
from numpy.linalg import LinAlgError
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, SpectralEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utils import efficient_deepcopy


# @njit(cache=True)
def batch_tournament_selection(individuals, k, tournsize=8, batch_size=8, fit_attr="fitness"):
    fitness_cases_num = len(individuals[0].case_values)
    idx_cases_batch = np.arange(0, fitness_cases_num)
    np.random.shuffle(idx_cases_batch)
    _batches = np.array_split(idx_cases_batch, max(fitness_cases_num // batch_size, 1))
    batch_ids = list(range(0, len(_batches)))
    assert len(_batches[0]) >= batch_size or fitness_cases_num < batch_size

    chosen = []
    while len(chosen) < k:
        batches: list = copy.deepcopy(batch_ids)
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


@njit(cache=True)
def selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, k):
    selected_individuals = []

    for i in range(k):
        candidates = list(range(len(case_values)))
        cases = np.arange(len(case_values[0]))
        np.random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = np.array([case_values[x][cases[0]] for x in candidates])
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median(np.array([abs(x - median_val) for x in errors_for_this_case]))
            if fit_weights > 0:
                best_val_for_case = np.max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list([x for x in candidates if case_values[x][cases[0]] >= min_val_to_survive])
            else:
                best_val_for_case = np.min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list([x for x in candidates if case_values[x][cases[0]] <= max_val_to_survive])
            cases = np.delete(cases, 0)
        selected_individuals.append(np.random.choice(np.array(candidates)))
    return selected_individuals


# @timeit
def selAutomaticEpsilonLexicaseFast(individuals, k):
    fit_weights = individuals[0].fitness.weights[0]
    case_values = np.array([ind.case_values for ind in individuals])
    index = selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, k)
    selected_individuals = [individuals[i] for i in index]
    return selected_individuals


def selRandomPlus(individuals, k, fit_attr="fitness"):
    # Select Top-2 individuals based on a random projected fitness value
    weight = np.random.random(len(individuals[0].case_values))
    return sorted(individuals, key=lambda x: np.sum(x.case_values * weight))[:k]


def selKnockout(individuals, k):
    cases = list(range(len(individuals[0].case_values)))
    final_pool = []
    for _ in range(k):
        random.shuffle(individuals)
        pool = list(range(len(individuals)))
        while len(pool) > 1:
            intermediate_pool = []
            for i in range(0, len(pool), 2):
                sample_index = random.choice(cases)
                if i + 1 >= len(pool) or \
                    (individuals[pool[i]].case_values[sample_index] <=
                     individuals[pool[i + 1]].case_values[sample_index]):
                    intermediate_pool.append(pool[i])
                else:
                    intermediate_pool.append(pool[i + 1])
            pool = intermediate_pool
        final_pool.append(individuals[pool[0]])
    return final_pool


def selBagging(individuals, k, fit_attr="fitness"):
    # Select Top-2 individuals based on bagging samples
    sample_indices = np.arange(len(individuals[0].case_values))
    bootstrap_indices = np.random.choice(sample_indices,
                                         size=sample_indices.shape[0],
                                         replace=True)
    return sorted(individuals, key=lambda x: np.sum(x.case_values[bootstrap_indices]))[:k]


def selTournamentNovelty(individuals, k, tournsize, fit_attr="fitness"):
    # Traditional Tournament: Select based on R2 score
    # Novelty Tournament: Select based on R2 score and Novelty score
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(min(aspirants, key=lambda x: np.sum(x.case_values)))
    return chosen


def selDoubleRound(individuals, k, count=5, base_operator='Lexicase', tournsize=3,
                   parsimonious=False):
    # First Round: Traditional Selection Operator
    # Second Round: Crossover
    selected_individuals = []
    for _ in range(k):
        if base_operator == 'Lexicase':
            inds = selAutomaticEpsilonLexicaseFast(individuals, count)
        elif base_operator == 'Tournament':
            inds = selTournamentNovelty(individuals, count, tournsize=tournsize)
        elif base_operator == 'Random':
            inds = selRandom(individuals, count)
        else:
            raise Exception
        base_ind = efficient_deepcopy(inds[0])
        for id in range(1, count):
            ind = inds[id]
            for i, g in enumerate(ind.gene):
                if (parsimonious and count * base_ind.coef[i] < ind.coef[i]) or \
                    (not parsimonious and base_ind.coef[i] < ind.coef[i]):
                    base_ind.coef[i] = ind.coef[i]
                    base_ind.gene[i] = copy.deepcopy(g)
            ## Only replace the worst feature by the best feature
            # min_index = min(range(len(ind.gene)), key=lambda x: base_ind.coef[x])
            # max_index = max(range(len(ind.gene)), key=lambda x: ind.coef[x])
            # if base_ind.coef[min_index] < ind.coef[max_index]:
            #     base_ind.coef[min_index] = ind.coef[max_index]
            #     base_ind.gene[min_index] = copy.deepcopy(ind.gene[max_index])
        del base_ind.fitness.values
        selected_individuals.append(base_ind)
    return selected_individuals


# @timeit
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


def selEpsilonLexicase(individuals, k, epsilon):
    selected_individuals = []

    for i in range(k):
        fit_weights = individuals[0].fitness.weights[0]
        candidates = individuals
        cases = list(range(len(individuals[0].case_values)))
        random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            if fit_weights > 0:
                best_val_for_case = max([x.case_values[cases[0]] for x in candidates])
                min_val_to_survive_case = best_val_for_case - epsilon
                candidates = list([x for x in candidates if x.case_values[cases[0]] >= min_val_to_survive_case])
            else:
                best_val_for_case = min([x.case_values[cases[0]] for x in candidates])
                max_val_to_survive_case = best_val_for_case + epsilon
                candidates = list([x for x in candidates if x.case_values[cases[0]] <= max_val_to_survive_case])

            cases.pop(0)

        selected_individuals.append(random.choice(candidates))
        # print('used cases', len(individuals[0].case_values) - len(cases))
        # print('remaining candidates', len(candidates))
    return selected_individuals


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def selMAPEliteClustering(individuals, old_list, map_elite_parameter, n_clusters=None):
    """
    :param individuals:
    :param old_list:
    :param map_elite_parameter:
    :param n_clusters:
    :return:
    elite_map: Selected individuals
    pop_pool: Store all individuals
    """
    if len(old_list) < len(individuals):
        map_dict = {}
        for id, ind in enumerate(individuals + old_list):
            map_dict[id] = ind
        return map_dict, list(map_dict.values())
    pop_size = len(individuals)
    # The individual pool may larger than the population size.
    # pool_ratio = map_elite_parameter.get('pool_ratio', 5)
    # pop_pool = list(sorted(individuals + old_list, key=lambda x: x.fitness.wvalues[0], reverse=True))
    # pop_pool = list({tuple(x.predicted_values): x for x in pop_pool}.values())
    # pop_pool = pop_pool[:pool_ratio * pop_size]
    # individuals = pop_pool
    individuals = individuals + old_list
    case_values = np.array([ind.predicted_values for ind in individuals])
    offspring_features = case_values

    fitness_ratio = map_elite_parameter.get('fitness_ratio', 0.2)
    mean_fitness = np.quantile([ind.fitness.wvalues[0] for ind in individuals], fitness_ratio)
    well_individual_values = np.array([ind.predicted_values for ind in
                                       filter(lambda x: x.fitness.wvalues[0] >= mean_fitness, individuals)])
    pca_dimension = map_elite_parameter.get('pca_dimension', 2)
    pca_dimension = min([pca_dimension, well_individual_values.shape[0], well_individual_values.shape[1]])

    # It is very important to find the reason of the singular value error
    try:
        pca = Pipeline(
            [('Scaler', StandardScaler()),
             ('PCA', PCA(n_components=pca_dimension))],
        ).fit(well_individual_values)
        offspring_features = pca.transform(offspring_features)
    except LinAlgError:
        pca = Pipeline(
            [('Scaler', StandardScaler()),
             ('PCA', PCA(n_components=pca_dimension, svd_solver='randomized'))],
        ).fit(well_individual_values)
        offspring_features = pca.transform(offspring_features)
    n_clusters = pop_size if n_clusters is None else n_clusters
    kmeans = Pipeline(
        [('Scaler', StandardScaler()),
         ('K-Means', KMeans(n_clusters=n_clusters))],
    )
    labels = kmeans.fit_predict(offspring_features)

    key_metric = map_elite_parameter['key_metric']
    map_dict = {}

    # Get the nearest point for each cluster center
    def calculate_size(ind: MultipleGeneGP):
        return np.sum([len(g) for g in ind.gene])

    for label, id in zip(labels, range(len(labels))):
        if label not in map_dict:
            map_dict[label] = individuals[id]
        elif key_metric == 'fitness' and individuals[id].fitness.wvalues >= map_dict[label].fitness.wvalues:
            map_dict[label] = individuals[id]
        elif key_metric == 'size' and calculate_size(individuals[id]) <= calculate_size(map_dict[label]):
            map_dict[label] = individuals[id]
    return map_dict, list(map_dict.values())


def selMAPElite(individuals, old_map, map_elite_parameter: dict):
    """
    """
    fitness_ratio = map_elite_parameter.get('fitness_ratio', 0.2)
    map_size = map_elite_parameter.get('map_size', 10)
    pca_dimension = map_elite_parameter.get('pca_dimension', 2)
    reduction_method = map_elite_parameter.get('reduction_method', 'PCA')

    plot = False
    individuals = individuals + list(old_map.values())
    individuals = list({tuple(x.predicted_values): x for x in individuals}.values())
    mean_fitness = np.quantile([ind.fitness.wvalues[0] for ind in individuals], fitness_ratio)
    # update a MAP
    map_dict = {}
    case_values = np.array([ind.predicted_values for ind in individuals])
    semantic_matrix = case_values
    if plot:
        plt.scatter(semantic_matrix[:, 0], semantic_matrix[:, 1],
                    c=np.array([ind.fitness.wvalues[0] for ind in individuals]))
        plt.show()
    well_individuals = [ind for ind in filter(lambda x: x.fitness.wvalues[0] >= mean_fitness, individuals)]
    well_individual_values = np.array([ind.predicted_values for ind in well_individuals])

    scaler = StandardScaler().fit(semantic_matrix)
    semantic_matrix = scaler.transform(semantic_matrix)
    well_individual_values = scaler.transform(well_individual_values)

    try:
        if reduction_method == 'PCA':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', PCA(n_components=pca_dimension))],
            ).fit(well_individual_values)
        elif reduction_method == 'KPCA(POLY)':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', KernelPCA(n_components=pca_dimension, kernel='poly'))],
            ).fit(well_individual_values)
        elif reduction_method == 'KPCA(RBF)':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', KernelPCA(n_components=pca_dimension, kernel='rbf'))],
            ).fit(well_individual_values)
        elif reduction_method == 'KPCA(SIGMOID)':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', KernelPCA(n_components=pca_dimension, kernel='sigmoid'))],
            ).fit(well_individual_values)
        elif reduction_method == 'KPCA(COSINE)':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', KernelPCA(n_components=pca_dimension, kernel='cosine'))],
            ).fit(well_individual_values)
        elif reduction_method == 'Isomap':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', Isomap(n_components=pca_dimension))],
            ).fit(well_individual_values)
        elif reduction_method == 'LLE':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', LocallyLinearEmbedding(n_components=pca_dimension))],
            ).fit(well_individual_values)
        elif reduction_method == 'MLLE':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', LocallyLinearEmbedding(n_components=pca_dimension, method='modified'))],
            ).fit(well_individual_values)
        elif reduction_method == 'HLLE':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', LocallyLinearEmbedding(n_neighbors=10, n_components=pca_dimension, method='hessian'))],
            ).fit(well_individual_values)
        elif reduction_method == 'LTSA':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', LocallyLinearEmbedding(n_components=pca_dimension, method='ltsa'))],
            ).fit(well_individual_values)
        elif reduction_method == 'TSNE':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', TSNE(n_components=pca_dimension))],
            ).fit(well_individual_values)
        elif reduction_method == 'SpectralEmbedding':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', SpectralEmbedding(n_components=pca_dimension))],
            ).fit(well_individual_values)
        else:
            raise Exception
    except Exception as e:
        if reduction_method == 'PCA':
            pca = Pipeline(
                [('Scaler', StandardScaler()),
                 ('PCA', PCA(n_components=pca_dimension, svd_solver='randomized'))],
            ).fit(well_individual_values)
        else:
            raise e
    if pca._can_transform():
        semantic_matrix = pca.transform(semantic_matrix)
    else:
        semantic_matrix = pca['PCA'].embedding_
    semantic_matrix = semantic_matrix[:, :pca_dimension + 1]

    for id in range(semantic_matrix.shape[1]):
        bins = np.linspace(semantic_matrix[:, id].min(), semantic_matrix[:, id].max(), map_size)
        semantic_matrix[:, id] = np.digitize(semantic_matrix[:, id], bins)

    pool = individuals if pca._can_transform() else well_individuals
    for id in range(len(pool)):
        case = tuple(int(x) for x in semantic_matrix[id])
        if case not in map_dict:
            map_dict[case] = pool[id]
        elif pool[id].fitness.wvalues >= map_dict[case].fitness.wvalues:
            map_dict[case] = pool[id]

    if plot:
        map_value = np.zeros((map_size, map_size))
        for k, v in map_dict.items():
            map_value[int(k[0]) - 1][int(k[1]) - 1] = v.fitness.wvalues[0]
        sns.heatmap(map_value, linewidth=0.5)
        plt.show()
    return map_dict


def selGPED(individuals, k, phi=None, rho=None):
    def selTournamentPlus(individuals, k, tournsize):
        """
        Select individuals based on case values
        """
        chosen = []
        for i in range(k):
            if len(individuals) > tournsize:
                aspirants = selRandom(individuals, tournsize)
            else:
                aspirants = individuals
            chosen.append(np.argmin(np.sum(aspirants, axis=1)))
        return chosen

    assert k % 2 == 0, 'k must be an even integer'
    inds = []
    case_values = np.array([ind.case_values for ind in individuals])
    if phi is None:
        phi, rho = get_semantic_matrix(case_values)

    for _ in range(k // 2):
        index = selTournamentPlus(case_values, 1, 7)
        parent_a = index[0]
        r = random.random()
        parent_b = None
        indexes = np.arange(0, len(phi))
        if r < 0.2:
            # high, opposite direction
            if phi[parent_a] <= 4:
                tmp_indexes = indexes[phi == (phi[parent_a] + 4)]
                if len(tmp_indexes) > 0:
                    parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                    parent_b = tmp_indexes[parent_b]
            else:
                tmp_indexes = indexes[phi == (phi[parent_a] - 4)]
                if len(tmp_indexes) > 0:
                    parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                    parent_b = tmp_indexes[parent_b]
        if r < 0.4 and parent_b is None:
            # median
            tmp_indexes = indexes[(phi != phi[parent_a]) & (rho != rho[parent_a])]
            if len(tmp_indexes) > 0:
                parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                parent_b = tmp_indexes[parent_b]
        if r < 0.8 and parent_b is None:
            # standard
            tmp_indexes = indexes[(phi != phi[parent_a]) | (rho != rho[parent_a])]
            if len(tmp_indexes) > 0:
                parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                parent_b = tmp_indexes[parent_b]
        if parent_b is None:
            # low
            tmp_indexes = indexes[(phi == phi[parent_a])]
            if len(tmp_indexes) > 0:
                parent_b = selTournamentPlus(case_values[tmp_indexes], 1, 7)[0]
                parent_b = tmp_indexes[parent_b]
        inds.append(individuals[parent_a])
        inds.append(individuals[parent_b])
    return inds


def get_semantic_matrix(case_values):
    semantic_matrix = case_values
    try:
        semantic_matrix = PCA(2).fit_transform(semantic_matrix)
    except:
        semantic_matrix = PCA(2, svd_solver='randomized').fit_transform(semantic_matrix)
    rho, phi = cart2pol(semantic_matrix[:, 0], semantic_matrix[:, 1])
    bins = np.linspace(rho.min(), rho.max(), 10)
    rho = np.digitize(rho, bins)
    assert np.all(rho <= 10) & np.all(rho > 0)
    bins = np.linspace(phi.min(), phi.max(), 8)
    phi = np.digitize(phi, bins)
    assert np.all(phi <= 8) & np.all(phi > 0)
    return phi, rho


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


def selHybrid(individuals, k):
    if 'LogisticRegression' in individuals[0].base_model:
        return selAutomaticEpsilonLexicaseFast(individuals, k)
    elif 'DT' in individuals[0].base_model:
        return selTournament(individuals, k, tournsize=7)
    else:
        raise Exception


if __name__ == '__main__':
    class MockIndividual():
        def __init__(self):
            self.case_values = np.random.randint(0, 2, 5)
            self.fitness = SimpleNamespace()
            self.fitness.weights = (1,)
            pass


    individuals = [MockIndividual() for _ in range(20)]
    pop = selAutomaticEpsilonLexicase(individuals, 1)
    print(pop)
