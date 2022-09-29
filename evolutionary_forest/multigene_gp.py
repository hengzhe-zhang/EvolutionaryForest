import copy
import random
import sys
import warnings
from functools import wraps
from inspect import isclass
from typing import Callable

import numpy as np
from deap.gp import PrimitiveTree, compile, cxOnePoint, mutUniform, mutShrink, mutInsert, cxOnePointLeafBiased, \
    PrimitiveSet
from deap.tools import selTournament, selRandom
from scipy.special import softmax
from scipy.stats import pearsonr, PearsonRConstantInputWarning, PearsonRNearConstantInputWarning
from tpot.base import TPOTBase

from evolutionary_forest.component.primitives import individual_to_tuple
from evolutionary_forest.component.syntax_tools import TransformerTool

warnings.simplefilter("ignore", category=PearsonRConstantInputWarning)
warnings.simplefilter("ignore", category=PearsonRNearConstantInputWarning)


def selTournamentFeature(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=lambda x: x[1]))
    return chosen


class MultipleGeneGP():
    def __init__(self, content, gene_num, basic_gene_num, pset=None,
                 tpot_model: TPOTBase = None, partition_scheme=None, base_model_list=None):
        self.gene = []
        # Some more basic features
        self.basic_gene = []
        self.gene_num = gene_num
        self.basic_gene_num = basic_gene_num
        self.active_gene_num = 0
        # This flag is only used for controlling the mutation and crossover
        self.basic_gene_mode = False
        for i in range(self.gene_num):
            self.gene.append(PrimitiveTree(content()))
        if isinstance(pset, MultiplePrimitiveSet) and self.basic_gene_num > 0:
            pset.change_state()
            for i in range(self.basic_gene_num):
                self.basic_gene.append(PrimitiveTree(content()))
            pset.change_state()
        if tpot_model != None:
            self.base_model = tpot_model._toolbox.individual()
        if base_model_list != None:
            self.base_model = random.choice(base_model_list.split(','))
        self.dynamic_leaf_size = random.randint(1, 10)
        self.dynamic_regularization = 1
        if partition_scheme is not None:
            # self.partition_scheme = np.random.randint(0, partition_scheme[1] , partition_scheme[0])
            # initialize partition scheme by decision tree
            if callable(partition_scheme):
                self.partition_scheme = partition_scheme()
            else:
                self.partition_scheme = partition_scheme
        else:
            self.partition_scheme = None
        # self-adaptive evolution
        # self.parameters = np.random.normal(0.5, 0.1, 2).clip(0.1, 0.9)

    def random_select_index(self):
        if self.basic_gene_mode:
            return random.randint(0, self.basic_gene_num - 1)
        else:
            return random.randint(0, self.gene_num - 1)

    def random_select(self):
        if self.basic_gene_mode:
            return self.basic_gene[random.randint(0, len(self.basic_gene) - 1)]
        else:
            return self.gene[random.randint(0, len(self.gene) - 1)]

    def tournament_selection(self, tournsize=2):
        """
        Current Issue:
        It may result in the loss of diversity
        """
        key = selTournamentFeature(list(enumerate(self.coef)), 1, tournsize=tournsize)
        return self.gene[key[0][0]]

    def best_gene(self):
        best_index = max(range(len(self.gene)), key=lambda x: np.abs(self.coef)[x])
        return self.gene[best_index]

    def worst_gene(self):
        worst_index = min(range(len(self.gene)), key=lambda x: np.abs(self.coef)[x])
        return self.gene[worst_index]

    def replace_worst_gene(self, gene):
        worst_index = min(range(len(self.gene)), key=lambda x: np.abs(self.coef)[x])
        self.gene[worst_index] = gene

    def softmax_selection(self, reverse=False):
        if reverse:
            weight = softmax(-20 * self.coef)
        else:
            weight = softmax(20 * self.coef)
        index = np.random.choice(np.arange(len(self.gene)), 1, p=weight)
        return self.gene[index[0]]

    def weighted_selection(self, reverse=False):
        weight = np.abs(self.coef)
        p = weight / np.sum(weight)
        if reverse:
            p[p == 0] = p[p > 0].min()
            p = 1 / p
            index = np.random.choice(np.arange(len(weight)), p=p / p.sum())
        else:
            index = np.random.choice(np.arange(len(weight)), p=p)
        return self.gene[index]

    def replace_weight_gene_inverse(self, gene):
        index = np.random.choice(np.arange(len(self.gene)), 1, p=softmax(-1 * np.abs(self.coef)))
        self.gene[index] = gene

    def deterministic_select(self):
        weight = np.abs(self.coef)
        return self.gene[np.argmax(-weight)]

    def __len__(self):
        return sum([len(g) for g in self.gene])

    def __str__(self):
        return str([str(g) for g in self.gene])

    def __repr__(self):
        return str([str(g) for g in self.gene])


def multiple_gene_evaluation(compiled_genes, x):
    result = []
    for gene in compiled_genes:
        result.append(gene(*x))
    return result


def multiple_gene_initialization(container, generator, gene_num, basic_gene_num, pset, tpot_model=None,
                                 partition_scheme=None, base_model_list=None):
    return container(generator, gene_num, basic_gene_num, pset, tpot_model, partition_scheme, base_model_list)


def multiple_gene_compile(expr: MultipleGeneGP, pset, basic_gene=False):
    """
    :param basic_gene: gene at the lower level, which could be used as ADF
    """
    gene_compiled = []
    if basic_gene:
        pset.change_state()
        for gene in expr.basic_gene:
            gene_compiled.append(compile(gene, pset))
        pset.change_state()
    else:
        if hasattr(expr, 'active_gene_num') and expr.active_gene_num > 0:
            # some genes may not be activated
            for gene in expr.gene[:expr.active_gene_num]:
                gene_compiled.append(compile(gene, pset))
        else:
            for gene in expr.gene:
                gene_compiled.append(compile(gene, pset))
    return gene_compiled


def pearson_check(x, y):
    if np.abs(pearsonr(x, y)[0]) < 0.95:
        return True
    else:
        return False


def cxOnePoint_multiple_gene_novelty(ind1: MultipleGeneGP, ind2: MultipleGeneGP, test_func):
    a = ind1.random_select()
    b = ind2.random_select()
    x, y = test_func(a), test_func(b)
    while True:
        cxOnePoint(a, b)
        z, w = test_func(a), test_func(b)
        if pearson_check(x, z) and pearson_check(y, z) and pearson_check(x, w) and pearson_check(y, w):
            break
    return ind1, ind2


def cxOnePoint_multiple_gene_diversity(ind1: MultipleGeneGP, ind2: MultipleGeneGP, visited_features: set):
    while True:
        a = ind1.random_select()
        b = ind2.random_select()
        cxOnePoint(a, b)
        if individual_to_tuple(a) in visited_features or individual_to_tuple(b) in visited_features:
            continue
    return ind1, ind2


def cxOnePoint_multiple_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.random_select(), ind2.random_select())
    return ind1, ind2


def cxOnePoint_multiple_gene_tournament(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    """
    A feature importance aware crossover operator based on the tournament selection
    """
    cxOnePoint(ind1.tournament_selection(), ind2.tournament_selection())
    return ind1, ind2


def cxOnePoint_multiple_gene_SC(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    """
    self-competitive crossover operator
    Using the worst individual as the base individual and migrate a portion of useful materials from well-behaved one
    """
    cxOnePoint(ind1.softmax_selection(reverse=True), copy.deepcopy(ind2.softmax_selection()))
    cxOnePoint(ind2.softmax_selection(reverse=True), copy.deepcopy(ind1.softmax_selection()))
    return ind1, ind2


def cxOnePoint_multiple_gene_same_index(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Crossover on the same location
    if ind1.basic_gene_mode:
        index = random.randint(0, ind1.basic_gene_num - 1)
        cxOnePoint(ind1.basic_gene[index], ind2.basic_gene[index])
    else:
        index = random.randint(0, ind1.gene_num - 1)
        cxOnePoint(ind1.gene[index], ind2.gene[index])
    return ind1, ind2


def cxOnePoint_multiple_gene_biased(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePointLeafBiased(ind1.random_select(), ind2.random_select(), 0.1)
    return ind1, ind2


def cxOnePoint_multiple_gene_same_weight(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Only cross features with same weight index
    index = random.randint(0, ind1.gene_num - 1)
    cxOnePoint(ind1.gene[np.argsort(ind1.coef)[index]], ind2.gene[np.argsort(ind2.coef)[index]])
    return ind1, ind2


def cxOnePoint_multiple_gene_best(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.best_gene(), ind2.best_gene())
    return ind1, ind2


def cxOnePoint_multiple_gene_worst(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.worst_gene(), ind2.worst_gene())
    return ind1, ind2


# extract all useful features from individuals
def extract_features(ind: MultipleGeneGP, useful=True, threshold=None):
    results = []
    if threshold is not None:
        mean_coef = threshold * np.mean(ind.coef)
    else:
        mean_coef = np.mean(ind.coef)
    for i, x in zip(ind.gene, ind.coef):
        if useful and x >= mean_coef:
            results.append(i)
        if not useful and x < mean_coef:
            results.append(i)
    return results


def feature_to_tuple(x):
    return tuple(a.name for a in x)


def cxOnePoint_multiple_gene_threshold(ind1: MultipleGeneGP, ind2: MultipleGeneGP, mutation: Callable,
                                       cross_pb: float, threshold: float):
    # extract all useful features from individuals for crossover
    useful_features_a, useful_features_b = extract_features(ind1, threshold=threshold), \
                                           extract_features(ind2, threshold=threshold)
    # replace useless features with useful features
    all_features = set(feature_to_tuple(x) for x in extract_features(ind1, False, threshold) +
                       extract_features(ind2, False, threshold))

    # print('Count Useful Features', len(useful_features_a) + len(useful_features_b))
    # print('Count Useless Features', len(all_features))

    def replace_useless_features(ind):
        # generated_features = set(feature_to_tuple(x) for x in extract_features(ind))
        generated_features = set()
        # replace useless feature with useful features
        for i in range(len(ind.gene)):
            variation = True
            # check whether a feature is a useless feature
            while feature_to_tuple(ind.gene[i]) in all_features or variation:
                variation = False
                a, b = random.choice(useful_features_a), random.choice(useful_features_b)
                # if random.random() < (len(useful_features_a) + len(useful_features_b)) / all_features:
                if random.random() < cross_pb:
                    gene1, gene2 = cxOnePoint(copy.deepcopy(a), copy.deepcopy(b))
                else:
                    gene1, gene2 = mutation(copy.deepcopy(a))[0], mutation(copy.deepcopy(b))[0]
                # ensure the generated features are not redundant features
                l = list(filter(lambda x: (feature_to_tuple(x) not in all_features)
                                          and
                                          (feature_to_tuple(x) not in generated_features)
                                , [gene1, gene2]))
                if len(l) > 0:
                    ind.gene[i] = random.choice(l)
                    generated_features.add(feature_to_tuple(ind.gene[i]))

    replace_useless_features(ind1)
    replace_useless_features(ind2)
    return ind1, ind2


def cxOnePoint_multiple_gene_cross(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    gene1 = copy.deepcopy(ind1.best_gene())
    gene2 = copy.deepcopy(ind2.best_gene())
    gene1, gene2 = cxOnePoint(gene1, gene2)
    ind1.replace_worst_gene(gene1)
    ind2.replace_worst_gene(gene2)
    return ind1, ind2


def cxOnePoint_all_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP, permutation=False):
    if permutation:
        a_list = np.random.permutation(np.arange(0, len(ind1.gene)))
        b_list = np.random.permutation(np.arange(0, len(ind2.gene)))
    else:
        a_list = np.arange(0, len(ind1.gene))
        b_list = np.arange(0, len(ind2.gene))
    for a, b in zip(a_list, b_list):
        cxOnePoint(ind1.gene[a], ind2.gene[b])
    return ind1, ind2


def cxOnePoint_all_gene_with_importance_probability(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Each tree has a probability to be varied
    probability = (ind1.coef + ind2.coef) / 2
    a_list = np.arange(0, len(ind1.gene))
    b_list = np.arange(0, len(ind2.gene))
    for i, a, b in zip(range(len(a_list)), a_list, b_list):
        if random.random() < probability[i]:
            cxOnePoint(ind1.gene[a], ind2.gene[b])
    return ind1, ind2


def cxOnePoint_all_gene_with_probability(ind1: MultipleGeneGP, ind2: MultipleGeneGP, probability):
    # Each tree has a probability to be varied
    a_list = np.arange(0, len(ind1.gene))
    b_list = np.arange(0, len(ind2.gene))
    for a, b in zip(a_list, b_list):
        if random.random() < probability:
            cxOnePoint(ind1.gene[a], ind2.gene[b])
            # cxOnePointLeafBiased(ind1.gene[a], ind2.gene[b], 0.1)
    return ind1, ind2


def mutUniform_multiple_gene_with_probability(individual: MultipleGeneGP, expr, pset, probability):
    for g in individual.gene:
        if random.random() < probability:
            mutUniform(g, expr, pset)
    return individual,


# automated crossover operator?
# 1. feature importance

def mutUniform_multiple_gene(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.random_select(), expr, pset)
    return individual,


def mutShrink_multiple_gene(individual: MultipleGeneGP, expr, pset):
    if random.random() < 0.5:
        mutUniform(individual.random_select(), expr, pset)
    else:
        mutShrink(individual.random_select())
    return individual,


# transformer-based mutation
def mutUniform_multiple_gene_transformer(individual: MultipleGeneGP, expr, pset,
                                         condition_probability: Callable[[], float],
                                         transformer: TransformerTool):
    if random.random() > condition_probability():
        return mutUniform_multiple_gene(individual, expr, pset)
    else:
        ind = transformer.sample(1)
        individual.replace_worst_gene(PrimitiveTree(ind[0]))
        return individual,


def mutUniform_multiple_gene_worst(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.worst_gene(), expr, pset)
    return individual,


def mutUniform_multiple_gene_threshold(individual: MultipleGeneGP, expr, pset):
    s = sum([1 - c for c in individual.coef])
    for g, c in zip(individual.gene, individual.coef):
        if random.random() < (1 - c) / s:
            mutUniform(individual.worst_gene(), expr, pset)
    return individual,


def mutUniform_multiple_gene_with_prob(individual: MultipleGeneGP, expr, pset, terminal_probs, primitive_probs):
    root_individual = individual
    individual = individual.random_select()
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_, terminal_probs=terminal_probs, primitive_probs=primitive_probs)
    return root_individual,


def mutInsert_multiple_gene(individual: MultipleGeneGP, pset):
    mutInsert(individual.random_select(), pset)
    return individual,


def cxOnePoint_multiple_gene_weighted(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Potential issue: This operator may overly cross important features and ignore useless features
    cxOnePoint(ind1.weighted_selection(), ind2.weighted_selection())
    return ind1, ind2


def mutWeight_multiple_gene(individual: MultipleGeneGP, expr, pset, threshold_ratio=0.2):
    good_features, threshold = construct_feature_pools([individual], True, threshold_ratio=threshold_ratio)

    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            positive = False
            if (positive and c >= threshold) or (not positive and c < threshold):
                new_features = mutUniform(copy.deepcopy(random.choice(good_features)), expr, pset)
                ind.gene[i] = new_features[0]

    replaces_features(individual)
    return individual,


def construct_feature_pools(pop, positive, threshold_ratio=0.2,
                            good_features_threshold=None):
    # positive: get all important features
    # negative: get all unimportant features
    good_features = []
    good_features_str = set()
    if good_features_threshold == None:
        threshold = np.quantile([ind.coef for ind in pop], threshold_ratio)
    elif good_features_threshold == 'mean':
        threshold = np.mean([ind.coef for ind in pop])
    else:
        # threshold for good features
        threshold = np.quantile([ind.coef for ind in pop], good_features_threshold)

    def add_features(ind):
        for c, x in zip(ind.coef, ind.gene):
            if (positive and c >= threshold) or (not positive and c < threshold):
                if str(x) in good_features_str:
                    continue
                # assign a fitness value for each feature
                x.fitness = c
                good_features.append(x)
                good_features_str.add(str(x))

    for ind in pop:
        add_features(ind)

    # calculate the threshold for crossover
    threshold = np.quantile([ind.coef for ind in pop], threshold_ratio)
    return good_features, threshold


def feature_crossover_cross(ind1, ind2, threshold_ratio):
    pop = [ind1, ind2]
    good_features, threshold = construct_feature_pools(pop, True, threshold_ratio=threshold_ratio)
    new_pop = []
    for ind in pop:
        ind = cxOnePoint_multiple_gene_weight_plus(ind, good_features, threshold, False)
        new_pop.append(ind)
    return new_pop


def feature_crossover_cross_global(ind1, ind2, regressor):
    pop = [ind1, ind2]
    new_pop = []
    for ind in pop:
        ind = cxOnePoint_multiple_gene_weight_plus(ind, regressor.good_features, regressor.cx_threshold, False)
        new_pop.append(ind)
    return new_pop


def feature_mutation_global(individual: MultipleGeneGP, expr, pset, regressor):
    threshold = regressor.cx_threshold

    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            if c < threshold:
                new_features = mutUniform(copy.deepcopy(random.choice(regressor.good_features)), expr, pset)
                ind.gene[i] = new_features[0]

    replaces_features(individual)
    return individual,


def pool_based_mutation(individual: MultipleGeneGP, expr, pset, regressor, pearson_selection=False,
                        feature_evaluation=None, tournament_size=0):
    # construct features from the pool of good features
    # in addition to that, we force new features are not equivalent to old features and useless features
    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            if str(ind.gene[i]) in regressor.generated_features or c < regressor.cx_threshold:
                # new_features = copy.deepcopy(ind.gene[i])
                while True:
                    def feature_selection():
                        if tournament_size == 0:
                            return random.choice(regressor.good_features)
                        else:
                            return selTournament(regressor.good_features, tournament_size, 1)[0]

                    if random.random() < 0.2:
                        # mutation (in order to deal with the case of infinite loop)
                        new_features = mutUniform(copy.deepcopy(feature_selection()),
                                                  expr, pset)[0]
                    else:
                        # crossover
                        new_features = cxOnePoint(copy.deepcopy(feature_selection()),
                                                  copy.deepcopy(feature_selection()))
                        new_features = random.choice(new_features)
                    regressor.repetitive_feature_count[-1] += 1
                    if not str(new_features) in regressor.generated_features:
                        # Pre-selection by Pearson correlation to ensure the synthesized feature is useful
                        # However, such a process might be misleading
                        if pearson_selection:
                            func = compile(new_features, pset)
                            Yp = result_calculation([func], regressor.X[:20], False).flatten()
                            if np.abs(pearsonr(Yp, regressor.y[:20])[0]) <= 0.05:
                                # useless features
                                regressor.generated_features.add(str(new_features))
                            else:
                                break
                        elif feature_evaluation != None:
                            # using semantic diversity when generating new features
                            y = feature_evaluation(new_features)
                            if not y in regressor.generated_features:
                                break
                        else:
                            break
                ind.gene[i] = new_features
            # regressor.generated_features.add(str(new_features))

    replaces_features(individual)
    return individual,


def feature_crossover(ind1, ind2, positive, threshold_ratio):
    pop = [ind1, ind2]
    good_features, threshold = construct_feature_pools(pop, positive, threshold_ratio=threshold_ratio)
    new_pop = []
    for ind in pop:
        ind = cxOnePoint_multiple_gene_weight_plus(ind, good_features, threshold, positive)
        new_pop.append(ind)
    return new_pop


def cxOnePoint_multiple_gene_weight_plus(ind: MultipleGeneGP, good_features, threshold, positive):
    def replaces_features(ind: MultipleGeneGP):
        for i, c in enumerate(ind.coef):
            if (positive and c >= threshold) or (not positive and c < threshold):
                new_features = cxOnePoint(copy.deepcopy(random.choice(good_features)),
                                          copy.deepcopy(random.choice(good_features)))
                ind.gene[i] = random.choice(new_features)

    replaces_features(ind)
    return ind


def mutUniform_multiple_gene_weighted(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.weighted_selection(), expr, pset)
    return individual,


def cxOnePoint_multiple_gene_deterministic(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.deterministic_select(), ind2.deterministic_select())
    return ind1, ind2


def mutUniform_multiple_gene_deterministic(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.deterministic_select(), expr, pset)
    return individual,


def staticLimit_multiple_gene(key, max_value):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind.gene) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                ind: MultipleGeneGP
                for j, x in enumerate(ind.gene):
                    if callable(max_value):
                        height_limitation = max_value()
                    else:
                        height_limitation = max_value

                    if key(x) > height_limitation:
                        # replace an unreasonable gene with a parent gene
                        # parent = keep_inds[i]
                        parent = random.choice(keep_inds)
                        gene = copy.deepcopy(parent[j])
                        ind.gene[j] = gene
                    assert key(ind.gene[j]) <= height_limitation
            return new_inds

        return wrapper

    return decorator


def result_calculation(func, data, original_features):
    result = multiple_gene_evaluation(func, data.T)
    result = result_post_process(result, data, original_features)
    return result


def result_post_process(result, data, original_features):
    # some post process step
    for i in range(len(result)):
        yp = result[i]
        if not isinstance(yp, np.ndarray):
            yp = np.full(len(data), 0)
        elif yp.size == 1:
            yp = np.full(len(data), yp)
        result[i] = yp
    if original_features:
        result = np.concatenate([np.array(result).T, data], axis=1)
    else:
        result = np.array(result).T
    result = np.nan_to_num(result, posinf=0, neginf=0)
    result = np.around(result, 15)
    return result


def genHalfAndHalf_with_prob(pset, min_, max_, terminal_probs, primitive_probs, type_=None):
    method = random.choice((genGrow_with_prob, genFull_with_prob))
    return method(pset, min_, max_, terminal_probs, primitive_probs, type_)


def genGrow_with_prob(pset, min_, max_, terminal_probs, primitive_probs, type_=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
               (depth >= min_ and random.random() < pset.terminalRatio)

    return generate_with_prob(pset, min_, max_, condition, terminal_probs, primitive_probs, type_)


def genFull_with_prob(pset, min_, max_, terminal_probs, primitive_probs, type_=None, sample_type=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate_with_prob(pset, min_, max_, condition, terminal_probs, primitive_probs, type_, sample_type)


def generate_with_prob(pset, min_, max_, condition, terminal_probs, primitive_probs, type_=None, sample_type=None):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                if sample_type == 'Dirichlet':
                    cat_prob = np.random.dirichlet(terminal_probs.flatten())
                    term = pset.terminals[type_][np.argmax(cat_prob)]
                else:
                    probability = terminal_probs.flatten()
                    probability = probability[:len(pset.terminals[type_])]
                    probability = probability / np.sum(probability)
                    term = np.random.choice(pset.terminals[type_], p=probability)
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add " \
                                 "a terminal of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = np.random.choice(pset.primitives[type_], p=primitive_probs.flatten())
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add " \
                                 "a primitive of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def varAndPlus(population, toolbox, cxpb, mutpb, gene_num, limitation_check):
    @limitation_check
    def mutation_function(*population):
        offspring = [toolbox.clone(ind) for ind in population]

        # Execute mutation and selection operator N-times
        for _ in range(gene_num):
            # Apply crossover and mutation on the offspring
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                                  offspring[i])
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                if random.random() < mutpb:
                    offspring[i], = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

        return offspring

    return mutation_function(*population)


class MultiplePrimitiveSet(PrimitiveSet):
    def __init__(self, name, arity, prefix="ARG"):
        super().__init__(name, arity, prefix)

    def terminal_backup(self):
        self.terminals_backup = copy.deepcopy(self.terminals)
        self.arguments_backup = copy.deepcopy(self.arguments)

    def change_state(self, flag=True):
        if flag:
            self.terminals, self.terminals_backup = self.terminals_backup, self.terminals
            self.arguments, self.arguments_backup = self.arguments_backup, self.arguments
