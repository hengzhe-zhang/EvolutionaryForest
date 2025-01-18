import copy
import math
import random
from abc import abstractmethod
from collections import Counter, defaultdict
from operator import attrgetter
from types import SimpleNamespace
from typing import TYPE_CHECKING, List

import numpy as np
import seaborn as sns
from deap.tools import selRandom, selTournament
from matplotlib import pyplot as plt
from numba import njit
from numpy.linalg import LinAlgError
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import EarlyStopping
from torch import optim
from torch.nn import MSELoss

from evolutionary_forest.component.evaluation import cos_sim
from evolutionary_forest.model.VAE import NeuralNetTransformer, VAE
from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utils import efficient_deepcopy

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class SelectionConfiguration:
    def __init__(self, **params):
        self.current_gen = 0


class Selection:
    @abstractmethod
    def select(self, individuals, k):
        pass


class MTLAutomaticLexicase(Selection):
    def __init__(self, number_of_tasks):
        self.number_of_tasks = number_of_tasks

    def select(self, individuals, k):
        fit_weights = individuals[0].fitness.weights[0]
        case_values = np.array([ind.case_values for ind in individuals])
        # Aggregate the case values into 'number_of_tasks' partitions
        aggregated_case_values = np.mean(
            case_values.reshape((len(individuals), -1, self.number_of_tasks)), axis=2
        )
        index, _ = selAutomaticEpsilonLexicaseNumba(
            aggregated_case_values, fit_weights, k
        )
        selected_individuals = [individuals[i] for i in index]
        return selected_individuals


class TournamentLexicase(Selection):
    def __init__(
        self,
        algorithm: "EvolutionaryForestRegressor",
        tournament_warmup_round=10,
        **kwargs,
    ):
        self.algorithm = algorithm
        self.tournament_warmup_round = tournament_warmup_round

    def select(self, individuals, k):
        # tournament as first several stages
        if self.tournament_warmup_round > 0 and self.tournament_warmup_round < 1:
            warmup_round = self.tournament_warmup_round * self.algorithm.n_gen
        else:
            warmup_round = self.tournament_warmup_round
        if self.algorithm.current_gen <= warmup_round:
            offspring = selTournament(individuals, k, tournsize=7)
        else:
            offspring = selAutomaticEpsilonLexicaseFast(individuals, k)
        return offspring


def hybrid_lexicase_dcd(individuals, k):
    inds = []
    for _ in range(k):
        if random.random() < 0.5:
            inds.append(selAutomaticEpsilonLexicaseFast(individuals, 1)[0])
        else:
            inds.append(selLexicaseDCD(individuals, 1)[0])
    return inds


# @njit(cache=True)
def batch_tournament_selection(
    individuals, k, tournsize=8, batch_size=8, fit_attr="fitness"
):
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
                cand_fitness_for_this_batch.append(
                    np.mean(idx.case_values[_batches[batches[0]]])
                )
            idx_winner = np.argmin(cand_fitness_for_this_batch)
            winner = idx_candidates[idx_winner]
            chosen.append(winner)
            batches.pop(0)
    return chosen


@njit(cache=True)
def sample_with_probability(elements, probabilities):
    cum_probs = np.cumsum(probabilities)
    random_num = np.random.random()
    for i in range(len(cum_probs)):
        if random_num < cum_probs[i]:
            return elements[i]
    return elements[-1]  # Fallback, in case of numerical issues


@njit(cache=True)
def median_axis_0(arr):
    result = np.empty(arr.shape[1])
    for i in range(arr.shape[1]):
        result[i] = np.median(arr[:, i])
    return result


@njit(cache=True)
def min_axis_0(arr):
    result = np.empty(arr.shape[1])
    for i in range(arr.shape[1]):
        result[i] = np.min(arr[:, i])
    return result


@njit(cache=True)
def selAutomaticEpsilonLexicaseCLNumba(case_values, fit_weights, k, inverse=False):
    selected_individuals = []
    avg_cases = 0

    for i in range(k):
        candidates = list(range(len(case_values)))
        cases = np.arange(len(case_values[0]))

        probability = min_axis_0(case_values)
        if not inverse:
            probability = max(probability) + min(probability) - probability

        while len(cases) > 0 and len(candidates) > 1:
            probability = probability / np.sum(probability)
            sample_index = sample_with_probability(np.arange(len(cases)), probability)
            sample_case = cases[sample_index]
            errors_for_this_case = np.array(
                [case_values[x][sample_case] for x in candidates]
            )
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median(
                np.array([abs(x - median_val) for x in errors_for_this_case])
            )
            if fit_weights > 0:
                best_val_for_case = np.max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list(
                    [
                        x
                        for x in candidates
                        if case_values[x][sample_case] >= min_val_to_survive
                    ]
                )
            else:
                best_val_for_case = np.min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list(
                    [
                        x
                        for x in candidates
                        if case_values[x][sample_case] <= max_val_to_survive
                    ]
                )
            cases = np.delete(cases, sample_index)
            probability = np.delete(probability, sample_index)
        avg_cases = (avg_cases * i + (len(case_values[0]) - len(cases))) / (i + 1)
        selected_individuals.append(np.random.choice(np.array(candidates)))
    return selected_individuals, avg_cases


def selAutomaticEpsilonLexicaseCLFast(individuals, k):
    fit_weights = individuals[0].fitness.weights[0]
    case_values = np.array([ind.case_values for ind in individuals])
    index, avg_cases = selAutomaticEpsilonLexicaseCLNumba(case_values, fit_weights, k)
    selected_individuals = [individuals[i] for i in index]
    return selected_individuals


def selAutomaticEpsilonLexicaseInverseCLFast(individuals, k):
    fit_weights = individuals[0].fitness.weights[0]
    case_values = np.array([ind.case_values for ind in individuals])
    # challenging example more probability
    index, avg_cases = selAutomaticEpsilonLexicaseCLNumba(
        case_values, fit_weights, k, inverse=True
    )
    selected_individuals = [individuals[i] for i in index]
    return selected_individuals


@njit(cache=True)
def selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, k):
    selected_individuals = []
    avg_cases = 0

    for i in range(k):
        candidates = list(range(len(case_values)))
        cases = np.arange(len(case_values[0]))
        np.random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = np.array(
                [case_values[x][cases[0]] for x in candidates]
            )
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median(
                np.array([abs(x - median_val) for x in errors_for_this_case])
            )
            if fit_weights > 0:
                best_val_for_case = np.max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list(
                    [
                        x
                        for x in candidates
                        if case_values[x][cases[0]] >= min_val_to_survive
                    ]
                )
            else:
                best_val_for_case = np.min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list(
                    [
                        x
                        for x in candidates
                        if case_values[x][cases[0]] <= max_val_to_survive
                    ]
                )
            cases = np.delete(cases, 0)
        avg_cases = (avg_cases * i + (len(case_values[0]) - len(cases))) / (i + 1)
        selected_individuals.append(np.random.choice(np.array(candidates)))
    return selected_individuals, avg_cases


def selLexicographicParsimonyPressure(individuals, k):
    individuals = list(sorted(individuals, key=lambda x: x.fitness.wvalues))
    selected = []
    r = 1 / 2
    for i in range(k):
        indices = [i for i in range(len(individuals))]
        rank = np.zeros(len(individuals))
        ix = 0
        cnt_rank = 0
        while ix < len(individuals):
            cnt = math.ceil(len(individuals) * r)
            rank[ix : ix + cnt] = cnt_rank
            ix += cnt
            r /= 2
            cnt_rank += 1
        a, b = selRandom(indices, 2)
        if rank[a] == rank[b] and len(individuals[a]) == len(individuals[b]):
            best_index = random.choice([a, b])
        elif rank[a] == rank[b]:
            best_index = min([a, b], key=lambda x: len(individuals[x]))
        else:
            best_index = max([a, b], key=lambda x: rank[x])
        selected.append(individuals[best_index])
    return selected


def plot_distribution(case_values, fit_weights):
    index, avg_cases = selAutomaticEpsilonLexicaseNumba(
        np.log1p(case_values), fit_weights, 1000
    )
    unique_indices, counts = np.unique(index, return_counts=True)
    print("Max cases", np.max(counts))
    plt.figure(figsize=(6, 3))
    plt.bar(unique_indices, counts, color="skyblue")
    plt.xlabel("Unique Indices")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of Selected Indices (Log)")

    plt.show()
    index, avg_cases = selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, 1000)
    unique_indices, counts = np.unique(index, return_counts=True)
    print("Max cases", np.max(counts))
    plt.figure(figsize=(6, 3))
    plt.bar(unique_indices, counts, color="skyblue")
    plt.xlabel("Unique Indices")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of Selected Indices (Normal)")
    plt.show()

    index, avg_cases = selAutomaticEpsilonLexicaseNumba(
        np.exp(case_values), fit_weights, 1000
    )
    unique_indices, counts = np.unique(index, return_counts=True)
    print("Max cases", np.max(counts))
    plt.figure(figsize=(6, 3))
    plt.bar(unique_indices, counts, color="skyblue")
    plt.xlabel("Unique Indices")
    plt.ylabel("Frequency")
    plt.title("Frequency Distribution of Selected Indices (Exp)")
    plt.show()


# @timeit
def selAutomaticEpsilonLexicaseFast(individuals, k, return_avg_cases=False):
    fit_weights = individuals[0].fitness.weights[0]
    case_values = np.array([ind.case_values for ind in individuals])
    index, avg_cases = selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, k)
    selected_individuals = [individuals[i] for i in index]
    if return_avg_cases:
        return selected_individuals, avg_cases
    else:
        return selected_individuals


def selGroupALS(individuals, k, inner_selection="Lexicase"):
    # Group individuals by base_model and count the occurrences
    count = Counter(ind.base_model for ind in individuals)

    # Create a dictionary to store individuals grouped by base_model
    groups = defaultdict(list)
    for ind in individuals:
        groups[ind.base_model].append(ind)

    # Calculate the total number of individuals
    total_individuals = sum(count.values())

    # Calculate the probability for each base_model based on group size
    probabilities = {model: size / total_individuals for model, size in count.items()}

    if inner_selection == "DCD-ALS":
        k *= 2
    # Sample k base models based on their probabilities
    sampled_models = random.choices(
        list(probabilities.keys()), weights=probabilities.values(), k=k
    )

    # Randomly select an individual from each of the sampled groups
    if inner_selection == "Lexicase":
        selected_individuals = [
            selAutomaticEpsilonLexicase(groups[model], 1)[0] for model in sampled_models
        ]
    elif inner_selection == "LexicaseDCD":
        # DCD on the same type
        selected_individuals = [
            selLexicaseDCD(groups[model], 1)[0] for model in sampled_models
        ]
    elif inner_selection == "DCD-ALS":
        # DCD on different types
        selected_individuals = [
            selAutomaticEpsilonLexicase(groups[model], 1)[0] for model in sampled_models
        ]
        selected_individuals = [
            tourn(selected_individuals[i], selected_individuals[i + 1])
            for i in range(0, len(selected_individuals), 2)
        ]
    else:
        raise Exception("Unsupported inner selection method")

    return selected_individuals


def selRandomPlus(individuals, k, fit_attr="fitness"):
    # Select Top-2 individuals based on a random projected fitness value
    weight = np.random.random(len(individuals[0].case_values))
    return sorted(individuals, key=lambda x: np.sum(x.case_values * weight))[:k]


def selKnockout(individuals, k, version="O", auto_case=False):
    """
    version:
    'O' represents only the one individual in each round
    'S' represents only the multiple individuals in each round
    """
    # np.unique([x.fitness.wvalues for x in final_pool],return_counts=True)
    cases = np.array(list(range(len(individuals[0].case_values))))
    final_pool = []
    number_of_cases = math.ceil(np.log2(len(individuals)))
    while len(final_pool) < k:
        random.shuffle(individuals)
        if auto_case:
            np.random.shuffle(cases)
            split_cases = np.array_split(cases, number_of_cases)
        pool = list(range(len(individuals)))
        round = 0
        while (version == "O" and len(pool) > 1) or (version == "S" and len(pool) > 2):
            intermediate_pool = []
            for i in range(0, len(pool), 2):
                if auto_case:
                    sample_index = split_cases[round]
                else:
                    sample_index = random.choice(cases)
                if i + 1 >= len(pool) or (
                    np.sum(individuals[pool[i]].case_values[sample_index])
                    <= np.sum(individuals[pool[i + 1]].case_values[sample_index])
                ):
                    intermediate_pool.append(pool[i])
                else:
                    intermediate_pool.append(pool[i + 1])
            pool = intermediate_pool
            round += 1
        for p in pool:
            final_pool.append(individuals[p])
    return final_pool


def selBagging(individuals, k, fit_attr="fitness"):
    # Select Top-2 individuals based on bagging samples
    sample_indices = np.arange(len(individuals[0].case_values))
    bootstrap_indices = np.random.choice(
        sample_indices, size=sample_indices.shape[0], replace=True
    )
    return sorted(individuals, key=lambda x: np.sum(x.case_values[bootstrap_indices]))[
        :k
    ]


def selTournamentNovelty(individuals, k, tournsize, fit_attr="fitness"):
    # Traditional Tournament: Select based on R2 score
    # Novelty Tournament: Select based on R2 score and Novelty score
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(min(aspirants, key=lambda x: np.sum(x.case_values)))
    return chosen


def selDoubleRound(
    individuals, k, count=5, base_operator="Lexicase", tournsize=3, parsimonious=False
):
    # First Round: Traditional Selection Operator
    # Second Round: Crossover
    selected_individuals = []
    for _ in range(k):
        if base_operator == "Lexicase":
            inds = selAutomaticEpsilonLexicaseFast(individuals, count)
        elif base_operator == "Tournament":
            inds = selTournamentNovelty(individuals, count, tournsize=tournsize)
        elif base_operator == "Random":
            inds = selRandom(individuals, count)
        else:
            raise Exception
        base_ind = efficient_deepcopy(inds[0])
        for id in range(1, count):
            ind = inds[id]
            for i, g in enumerate(ind.gene):
                if (parsimonious and count * base_ind.coef[i] < ind.coef[i]) or (
                    not parsimonious and base_ind.coef[i] < ind.coef[i]
                ):
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
            median_absolute_deviation = np.median(
                [abs(x - median_val) for x in errors_for_this_case]
            )
            if fit_weights > 0:
                best_val_for_case = np.max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list(
                    [
                        x
                        for x in candidates
                        if x.case_values[cases[0]] >= min_val_to_survive
                    ]
                )
            else:
                best_val_for_case = np.min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list(
                    [
                        x
                        for x in candidates
                        if x.case_values[cases[0]] <= max_val_to_survive
                    ]
                )
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
                candidates = list(
                    [
                        x
                        for x in candidates
                        if x.case_values[cases[0]] >= min_val_to_survive_case
                    ]
                )
            else:
                best_val_for_case = min([x.case_values[cases[0]] for x in candidates])
                max_val_to_survive_case = best_val_for_case + epsilon
                candidates = list(
                    [
                        x
                        for x in candidates
                        if x.case_values[cases[0]] <= max_val_to_survive_case
                    ]
                )

            cases.pop(0)

        selected_individuals.append(random.choice(candidates))
        # print('used cases', len(individuals[0].case_values) - len(cases))
        # print('remaining candidates', len(candidates))
    return selected_individuals


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
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

    fitness_ratio = map_elite_parameter.get("fitness_ratio", 0.2)
    mean_fitness = np.quantile(
        [ind.fitness.wvalues[0] for ind in individuals], fitness_ratio
    )
    well_individual_values = np.array(
        [
            ind.predicted_values
            for ind in filter(
                lambda x: x.fitness.wvalues[0] >= mean_fitness, individuals
            )
        ]
    )
    pca_dimension = map_elite_parameter.get("pca_dimension", 2)
    pca_dimension = min(
        [
            pca_dimension,
            well_individual_values.shape[0],
            well_individual_values.shape[1],
        ]
    )

    # It is very important to find the reason of the singular value error
    try:
        pca = Pipeline(
            [("Scaler", StandardScaler()), ("PCA", PCA(n_components=pca_dimension))],
        ).fit(well_individual_values)
        offspring_features = pca.transform(offspring_features)
    except LinAlgError:
        pca = Pipeline(
            [
                ("Scaler", StandardScaler()),
                ("PCA", PCA(n_components=pca_dimension, svd_solver="randomized")),
            ],
        ).fit(well_individual_values)
        offspring_features = pca.transform(offspring_features)
    n_clusters = pop_size if n_clusters is None else n_clusters
    kmeans = Pipeline(
        [("Scaler", StandardScaler()), ("K-Means", KMeans(n_clusters=n_clusters))],
    )
    labels = kmeans.fit_predict(offspring_features)

    key_metric = map_elite_parameter["key_metric"]
    map_dict = {}

    # Get the nearest point for each cluster center
    def calculate_size(ind: MultipleGeneGP):
        return np.sum([len(g) for g in ind.gene])

    for label, id in zip(labels, range(len(labels))):
        if label not in map_dict:
            map_dict[label] = individuals[id]
        elif (
            key_metric == "fitness"
            and individuals[id].fitness.wvalues >= map_dict[label].fitness.wvalues
        ):
            map_dict[label] = individuals[id]
        elif key_metric == "size" and calculate_size(individuals[id]) <= calculate_size(
            map_dict[label]
        ):
            map_dict[label] = individuals[id]
    return map_dict, list(map_dict.values())


def selMAPElites(
    individuals,
    old_map,
    map_elite_parameter: dict,
    target: np.ndarray = None,
    data_augmentation=0,
):
    """ """
    fitness_ratio = map_elite_parameter.get("fitness_ratio", 0.2)
    map_size = map_elite_parameter.get("map_size", 10)
    pca_dimension = map_elite_parameter.get("pca_dimension", 2)
    reduction_method = map_elite_parameter.get("reduction_method", "PCA")
    both_real_and_synthetic = map_elite_parameter.get("both_real_and_synthetic", False)
    plot_function = map_elite_parameter.get("plot_function", None)

    plot = False
    individuals = individuals + list(old_map.values())
    individuals = list({tuple(x.predicted_values): x for x in individuals}.values())
    mean_fitness = np.quantile(
        [ind.fitness.wvalues[0] for ind in individuals], fitness_ratio
    )
    # update a MAP
    elites_dict = {}
    behavior_space = np.array([ind.predicted_values for ind in individuals])
    # extract behavior of all good individuals
    well_individuals = [
        ind
        for ind in filter(lambda x: x.fitness.wvalues[0] >= mean_fitness, individuals)
    ]
    if data_augmentation > 0:
        original_semantic_space = np.array(
            [ind.predicted_values for ind in well_individuals]
        )
        synthetic_semantic_space = np.concatenate(
            [
                (1 - data_augmentation) * original_semantic_space
                + data_augmentation * target,
                (data_augmentation - 1) * original_semantic_space
                + (2 - data_augmentation) * target,
            ],
            axis=0,
        )
    else:
        original_semantic_space = None
        synthetic_semantic_space = np.array(
            [ind.predicted_values for ind in well_individuals]
        )

    behavior_space -= target
    synthetic_semantic_space -= target
    scaler = StandardScaler(with_mean=False)
    scaler.fit(synthetic_semantic_space)
    behavior_space = scaler.transform(behavior_space)
    synthetic_semantic_space = scaler.transform(synthetic_semantic_space)
    if data_augmentation > 0:
        # transform augmented data
        original_semantic_space -= target
        original_semantic_space = scaler.transform(original_semantic_space)

    def common_tsne(metric="euclidean", init="pca"):
        return TSNE(
            n_components=pca_dimension,
            perplexity=30 if len(synthetic_semantic_space) >= 50 else 5,
            metric=metric,
            init=init,
        )

    try:
        if reduction_method == "PCA":
            pca = PCA(n_components=pca_dimension)
        elif reduction_method == "KPCA(POLY)":
            pca = KernelPCA(n_components=pca_dimension, kernel="poly")
        elif reduction_method == "KPCA(RBF)":
            pca = KernelPCA(n_components=pca_dimension, kernel="rbf")
        elif reduction_method == "KPCA(SIGMOID)":
            pca = KernelPCA(n_components=pca_dimension, kernel="sigmoid")
        elif reduction_method == "KPCA(COSINE)":
            pca = KernelPCA(n_components=pca_dimension, kernel="cosine")
        elif reduction_method == "KPCA(COSINE)+TSNE":
            pca = Pipeline(
                [("PCA-Preprocess", KernelPCA(kernel="cosine")), ("PCA", common_tsne())]
            )
        elif reduction_method == "TSNE(Cosine)":
            pca = common_tsne(metric="cosine")
        elif reduction_method == "Isomap(Cosine)":
            pca = Isomap(metric="cosine")
        elif reduction_method == "Isomap":
            pca = Isomap(n_components=pca_dimension)
        elif reduction_method == "LLE":
            pca = LocallyLinearEmbedding(n_components=pca_dimension)
        elif reduction_method == "MLLE":
            pca = LocallyLinearEmbedding(n_components=pca_dimension, method="modified")
        elif reduction_method == "HLLE":
            pca = LocallyLinearEmbedding(
                n_neighbors=10, n_components=pca_dimension, method="hessian"
            )
        elif reduction_method == "LTSA":
            pca = LocallyLinearEmbedding(n_components=pca_dimension, method="ltsa")
        elif reduction_method == "TSNE":
            pca = common_tsne()
        elif reduction_method == "PCA-TSNE":
            pca = Pipeline(
                [("PCA-Preprocess", PCA(n_components=0.99)), ("PCA", common_tsne())]
            )
        elif reduction_method == "UMAP":
            from umap import UMAP

            pca = UMAP(n_components=pca_dimension)
        elif reduction_method == "SpectralEmbedding":
            pca = SpectralEmbedding(n_components=pca_dimension)
        elif reduction_method == "Beta-VAE":
            pca = NeuralNetTransformer(
                VAE,
                criterion=MSELoss(),
                optimizer=optim.Adam,
                module__input_unit=behavior_space.shape[1],
                max_epochs=2000,
                callbacks=[EarlyStopping(patience=50)],
                verbose=False,
            )
        else:
            print("Unsupported method", reduction_method)
            raise Exception
        pca.fit(synthetic_semantic_space.astype(np.float32))
    except Exception as e:
        if reduction_method == "PCA":
            pca = PCA(n_components=pca_dimension, svd_solver="randomized")
        elif reduction_method == "PCA-TSNE":
            pca = Pipeline(
                [
                    ("PCA-Preprocess", PCA(0.99, svd_solver="randomized")),
                    ("PCA", common_tsne(init="random")),
                ]
            )
        elif reduction_method == "KPCA(COSINE)+TSNE":
            pca = Pipeline(
                [
                    ("PCA-Preprocess", KernelPCA(kernel="cosine")),
                    ("PCA", common_tsne(init="random")),
                ]
            )
        elif reduction_method == "TSNE(Cosine)":
            pca = common_tsne(metric="cosine", init="random")
        elif reduction_method == "LLE":
            pca = LocallyLinearEmbedding(
                eigen_solver="dense", n_components=pca_dimension
            )
        elif reduction_method == "MLLE":
            pca = LocallyLinearEmbedding(
                eigen_solver="dense", n_components=pca_dimension, method="modified"
            )
        elif reduction_method == "HLLE":
            pca = LocallyLinearEmbedding(
                eigen_solver="dense",
                n_neighbors=10,
                n_components=pca_dimension,
                method="hessian",
            )
        elif reduction_method == "LTSA":
            pca = LocallyLinearEmbedding(
                eigen_solver="dense", n_components=pca_dimension, method="ltsa"
            )
        elif reduction_method == "TSNE":
            pca = TSNE(
                n_components=pca_dimension,
                perplexity=30 if len(synthetic_semantic_space) >= 50 else 5,
                init="random",
            )
        else:
            print("Unsupported reduction algorithm", reduction_method)
            raise e
        pca.fit(synthetic_semantic_space.astype(np.float32))
    if (isinstance(pca, Pipeline) and pca._can_transform()) or (
        not isinstance(pca, Pipeline) and hasattr(pca, "transform")
    ):
        if data_augmentation > 0:
            # Only transform original data. It means discretization is on original data.
            # Here, one question exist, should we transform on original data.
            # If we transform on original data, we will definitely get mismatch distribution.
            if both_real_and_synthetic:
                # this is to test whether to use synthetic data in discretization
                behavior_space = pca.transform(
                    synthetic_semantic_space.astype(np.float32)
                )
            else:
                # this is to validate the situation of mismatch distribution
                behavior_space = pca.transform(
                    original_semantic_space.astype(np.float32)
                )
        else:
            # If no synthetic data, can directly perform on synthetic space.
            behavior_space = pca.transform(synthetic_semantic_space.astype(np.float32))
    else:
        # One problem here is that get all data will have an influence on discretization. Be Careful!
        if isinstance(pca, Pipeline):
            behavior_space = np.copy(pca[-1].embedding_)
        else:
            # Get embedding directly
            behavior_space = np.copy(pca.embedding_)
    behavior_space = behavior_space[:, : pca_dimension + 1]
    # behavior space before discretization
    original_behavior_space = np.copy(behavior_space)

    # discretize behavior space
    for id in range(behavior_space.shape[1]):
        bins = np.linspace(
            behavior_space[:, id].min(), behavior_space[:, id].max(), map_size + 1
        )
        behavior_space[:, id] = np.digitize(behavior_space[:, id], bins)
        data = behavior_space[:, id]
        # for the boundary, it should belong to the boundary class
        data[data == map_size + 1] = map_size
        assert np.all(data <= map_size)

    # save good individuals
    pool = well_individuals
    elites_behavior_dict = {}
    for id in range(len(pool)):
        case = tuple(int(x) for x in behavior_space[id])
        if case not in elites_dict:
            elites_dict[case] = pool[id]
            elites_behavior_dict[case] = original_behavior_space[id]
        elif pool[id].fitness.wvalues >= elites_dict[case].fitness.wvalues:
            elites_dict[case] = pool[id]
            elites_behavior_dict[case] = original_behavior_space[id]
    if plot_function is not None:
        plot_function(
            reduction_method,
            original_behavior_space,
            well_individuals,
            elites_dict,
            elites_behavior_dict,
        )
    if plot:
        map_value = np.zeros((map_size, map_size))
        for k, v in elites_dict.items():
            map_value[int(k[0]) - 1][int(k[1]) - 1] = v.fitness.wvalues[0]
        sns.heatmap(map_value, linewidth=0.5)
        plt.show()
    return elites_dict


def selMaxAngleSelection(individuals: list, k, target: np.ndarray, unique=False):
    """
    A variant of ADS in Qi Chen's paper
    :param individuals: Individuals from the current population
    :param k: Number of individuals to be selected
    :param target: The target semantics
    :return: Parent individuals
    """
    assert k % 2 == 0
    individual_list = []
    for _ in range(k):
        if len(individuals) < 2:
            break
        all_case_values = np.array(
            [ind.predicted_values - target for ind in individuals]
        )
        parent_a = selTournament(individuals, 1, tournsize=7)[0]
        best_one = np.argmin(
            cosine_similarity(parent_a.case_values.reshape(1, -1), all_case_values)[0]
        )
        parent_b = individuals[best_one]
        if parent_a == parent_b:
            break
        individual_list.append(parent_a)
        individual_list.append(parent_b)
        if unique:
            individuals.remove(parent_a)
            individuals.remove(parent_b)
    return individual_list


def selAngleDrivenSelection(individuals, k, target: np.ndarray, unique=False):
    """
    Original ADS in Qi Chen's paper
    :param individuals: Individuals from the current population
    :param k: Number of individuals to be selected
    :param target: The target semantics
    :return: Parent individuals
    """
    assert k % 2 == 0
    individual_list = []
    for _ in range(k):
        if len(individuals) < 2:
            break
        parent_a = selTournament(individuals, 1, tournsize=7)[0]
        parent_b = None
        n_trails = 10
        threshold_angle = 0
        best_angle = 1
        for i in range(n_trails):
            candidate = selTournament(individuals, 1, tournsize=7)[0]
            a_values = parent_a.case_values - target
            b_values = candidate.case_values - target
            similarity = cosine_similarity(
                a_values.reshape(1, -1), b_values.reshape(1, -1)
            )[0][0]
            if similarity < threshold_angle:
                parent_b = candidate
                break
            else:
                if similarity < best_angle:
                    best_angle = similarity
                    parent_b = candidate
        if parent_b is None:
            break
        individual_list.append(parent_a)
        individual_list.append(parent_b)
        if unique:
            individuals.remove(parent_a)
            individuals.remove(parent_b)
    return individual_list


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

    assert k % 2 == 0, "k must be an even integer"
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
        semantic_matrix = PCA(2, svd_solver="randomized").fit_transform(semantic_matrix)
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
        median_absolute_deviation = np.median(
            [abs(x - median_val) for x in errors_for_this_case]
        )
        if fit_weights > 0:
            best_val_for_case = np.max(errors_for_this_case)
            min_val_to_survive = best_val_for_case - median_absolute_deviation
            c = list(
                [x for x in candidates if x.case_values[cases[0]] >= min_val_to_survive]
            )
            if len(c) < k:
                break
            candidates = c
        else:
            best_val_for_case = np.min(errors_for_this_case)
            max_val_to_survive = best_val_for_case + median_absolute_deviation
            c = list(
                [x for x in candidates if x.case_values[cases[0]] <= max_val_to_survive]
            )
            if len(c) < k:
                break
            candidates = c
        cases.pop(0)
    return random.sample(candidates, k)


def selStatisticsTournament(individuals, k, tournsize):
    """
    :param individuals: population
    :param k: number of offspring
    :param tournsize: tournament size
    """
    chosen = []
    for i in range(k):
        a = random.choice(individuals)
        for _ in range(tournsize):
            b = random.choice(individuals)
            if np.all(np.equal(a.case_values, b.case_values)):
                continue
            p_value = wilcoxon(a.case_values, b.case_values).pvalue
            if p_value < 0.05:
                a = max([a, b], key=lambda x: x.fitness.wvalues)
            else:
                a = min([a, b], key=lambda x: len(x))
        chosen.append(a)
    return chosen


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
    if "LogisticRegression" in individuals[0].base_model:
        return selAutomaticEpsilonLexicaseFast(individuals, k)
    elif "DT" in individuals[0].base_model:
        return selTournament(individuals, k, tournsize=7)
    else:
        raise Exception


def selRoulette(individuals, k, fit_attr="fitness"):
    """
    It should be weighted values
    """
    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    sum_fits = sum(getattr(ind, fit_attr).wvalues[0] for ind in individuals)
    chosen = []
    for i in range(k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in s_inds:
            sum_ += getattr(ind, fit_attr).wvalues[0]
            if sum_ > u:
                chosen.append(ind)
                break

    return chosen


def tourn(ind1, ind2):
    if ind1.fitness.dominates(ind2.fitness):
        return ind1
    elif ind2.fitness.dominates(ind1.fitness):
        return ind2

    if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
        return ind2
    elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
        return ind1

    if random.random() <= 0.5:
        return ind1
    return ind2


def selLexicaseDCD(individuals: List[MultipleGeneGP], k):
    if k > len(individuals):
        raise ValueError(
            "selTournamentDCD: k must be less than or equal to individuals length"
        )

    if k == len(individuals) and k % 4 != 0:
        raise ValueError(
            "selTournamentDCD: k must be divisible by four if k == len(individuals)"
        )

    # individuals = adaptive_factors_elimination(individuals)

    individuals_chosen = selAutomaticEpsilonLexicaseFast(individuals, k * 2)

    chosen = []
    for i in range(0, k * 2, 2):
        chosen.append(tourn(individuals_chosen[i], individuals_chosen[i + 1]))
    return chosen


def adaptive_factors_elimination(individuals):
    if any(
        [
            p.individual_configuration.dynamic_standardization is not None
            for p in individuals
        ]
    ):
        base = np.random.choice(
            [p.individual_configuration.dynamic_standardization for p in individuals]
        )
        individuals = [
            p
            for p in individuals
            if type(p.individual_configuration.dynamic_standardization) == type(base)
        ]
    return individuals


def selLexicaseKNN(
    individuals, k, base_operator="Lexicase", neighbor=3, strategy="Random", y=None
):
    chosen = []
    for i in range(0, k, 2):
        if base_operator == "Lexicase":
            a: MultipleGeneGP = selAutomaticEpsilonLexicaseFast(individuals, 1)[0]
        elif base_operator.startswith("Tournament"):
            tournsize = int(base_operator.split("~")[1])
            a: MultipleGeneGP = selTournament(individuals, 1, tournsize=tournsize)[0]
        else:
            raise Exception

        # Calculate distances in semantic space
        distances = []
        for ind in individuals:
            if np.all(np.array(a.fitness.wvalues) >= np.array(ind.fitness.wvalues)):
                # dominating
                continue
            if "-C" in strategy:
                dist = cos_sim(a.predicted_values - y, ind.predicted_values - y)
            else:
                dist = np.linalg.norm(a.predicted_values - ind.predicted_values)
            if dist == 0:
                # equivalent semantics
                continue
            distances.append((dist, ind))

        if len(distances) == 0:
            continue
        if strategy == "Lexicase":
            pool = [ind for _, ind in distances]
            chosen.append(a)
            chosen.append(selAutomaticEpsilonLexicaseFast(pool, 1)[0])
            continue
        if strategy == "Tournament":
            pool = [ind for _, ind in distances]
            chosen.append(a)
            chosen.append(selTournament(pool, 1, neighbor)[0])
            continue

        # Sort individuals based on distance
        if "Max" in strategy:
            # default prefer near individuals
            distances.sort(key=lambda x: -x[0])
        elif "Min" in strategy:
            distances.sort(key=lambda x: x[0])
        elif "Random" in strategy:
            random.shuffle(distances)
        else:
            raise Exception

        # Choose the k-nearest neighbors
        neighbors = [ind for _, ind in distances[:neighbor]]

        # best sharpness
        chosen.append(a)
        if "Random" in strategy:
            chosen.append(random.choice(neighbors))
        elif "BestSharpness" in strategy:
            chosen.append(max(neighbors, key=lambda x: x.fitness.wvalues[1]))
        elif "BestAccuracy" in strategy:
            chosen.append(max(neighbors, key=lambda x: x.fitness.wvalues[0]))
        elif "BestSum" in strategy:
            chosen.append(
                max(
                    neighbors, key=lambda x: x.fitness.wvalues[0] + x.fitness.wvalues[1]
                )
            )
        else:
            raise Exception
    return chosen


def selHOFRandom(individuals, k, hof):
    return random.sample(list(hof), k)


def selLexicaseTournament(individuals, k):
    # First lexicase, then tournament with a size of 2
    individuals_chosen = selAutomaticEpsilonLexicaseFast(individuals, k * 2)

    chosen = []
    for i in range(0, k * 2, 2):
        # larger is better
        if (
            individuals_chosen[i].fitness.wvalues
            > individuals_chosen[i + 1].fitness.wvalues
        ):
            chosen.append((individuals_chosen[i]))
        else:
            chosen.append((individuals_chosen[i + 1]))
            assert (
                individuals_chosen[i].fitness.wvalues
                <= individuals_chosen[i + 1].fitness.wvalues
            )
    assert len(chosen) == k
    return chosen


class MockIndividual:
    def __init__(self, case_values):
        self.case_values = case_values
        self.fitness = SimpleNamespace()
        self.fitness.weights = (1,)


if __name__ == "__main__":
    individuals = [MockIndividual(np.random.randint(0, 2, 5)) for _ in range(20)]
    pop = selAutomaticEpsilonLexicase(individuals, 1)
    print(pop)
