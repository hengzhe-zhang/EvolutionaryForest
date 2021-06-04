import copy
import random
import sys
from functools import wraps
from inspect import isclass

import numpy as np
from deap.gp import PrimitiveTree, compile, cxOnePoint, mutUniform, mutShrink, mutInsert
from scipy.special import softmax


class MultipleGeneGP():
    def __init__(self, content, gene_num):
        self.gene = []
        self.gene_num = gene_num
        for i in range(self.gene_num):
            self.gene.append(PrimitiveTree(content()))

    def random_select(self):
        return self.gene[random.randint(0, self.gene_num - 1)]

    def weight_select(self, positive=True):
        weight = np.abs(self.coef)
        if positive:
            p = softmax(abs(weight))
        else:
            p = softmax(-abs(weight))
        index = np.random.choice(np.arange(len(weight)), p=p)
        return self.gene[index]

    def deterministic_select(self):
        weight = np.abs(self.coef)
        return self.gene[np.argmax(-weight)]

    def __len__(self):
        return sum([len(g) for g in self.gene])


def multiple_gene_evaluation(compiled_genes, x):
    result = []
    for gene in compiled_genes:
        result.append(gene(*x))
    return result


def multiple_gene_initialization(container, generator, gene_num=5):
    return container(generator, gene_num)


def multiple_gene_compile(expr: MultipleGeneGP, pset):
    gene_compiled = []
    for gene in expr.gene:
        gene_compiled.append(compile(gene, pset))
    return gene_compiled


def cxOnePoint_multiple_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.random_select(), ind2.random_select())
    return ind1, ind2


def cxOnePoint_multiple_all_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP, permutation=False):
    if permutation:
        a_list = np.random.permutation(np.arange(0, len(ind1.gene)))
        b_list = np.random.permutation(np.arange(0, len(ind2.gene)))
    else:
        a_list = np.arange(0, len(ind1.gene))
        b_list = np.arange(0, len(ind2.gene))
    for a, b in zip(a_list, b_list):
        cxOnePoint(ind1.gene[a], ind2.gene[b])
    return ind1, ind2


def mutUniform_multiple_gene(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.random_select(), expr, pset)
    return individual,


def mutUniform_multiple_gene_with_prob(individual: MultipleGeneGP, expr, pset, terminal_probs, primitive_probs):
    root_individual = individual
    individual = individual.random_select()
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_, terminal_probs=terminal_probs, primitive_probs=primitive_probs)
    return root_individual,


def mutShrink_multiple_gene(individual: MultipleGeneGP):
    mutShrink(individual.random_select())
    return individual,


def mutInsert_multiple_gene(individual: MultipleGeneGP, pset):
    mutInsert(individual.random_select(), pset)
    return individual,


def cxOnePoint_multiple_gene_weight(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    cxOnePoint(ind1.weight_select(), ind2.weight_select())
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


def construct_feature_pools(pop, positive, threshold_ratio=0.2):
    # positive: get all important features
    # negative: get all unimportant features
    good_features = []
    # threshold = np.mean([ind.coef for ind in pop])
    threshold = np.quantile([ind.coef for ind in pop], threshold_ratio)

    def add_features(ind):
        for c, x in zip(ind.coef, ind.gene):
            if (positive and c >= threshold) or (not positive and c < threshold):
                good_features.append(x)

    for ind in pop:
        add_features(ind)
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


def mutUniform_multiple_gene_weight(individual: MultipleGeneGP, expr, pset):
    mutUniform(individual.weight_select(), expr, pset)
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
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                limit_exceed = False
                for x in ind.gene:
                    if key(x) > max_value:
                        limit_exceed = True
                        break
                if limit_exceed:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds

        return wrapper

    return decorator


def result_calculation(func, data, original_features):
    result = multiple_gene_evaluation(func, data.T)
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
    result = np.nan_to_num(result)
    return result


def genFull_with_prob(pset, min_, max_, terminal_probs, primitive_probs, type_=None):
    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate_with_prob(pset, min_, max_, condition, terminal_probs, primitive_probs, type_)


def generate_with_prob(pset, min_, max_, condition, terminal_probs, primitive_probs, type_=None):
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = np.random.choice(pset.terminals[type_], p=terminal_probs.flatten())
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
