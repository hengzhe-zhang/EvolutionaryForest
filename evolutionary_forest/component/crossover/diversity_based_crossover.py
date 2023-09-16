import numpy as np
from deap.gp import cxOnePoint
from scipy.stats import pearsonr

from evolutionary_forest.component.primitives import individual_to_tuple
from evolutionary_forest.multigene_gp import MultipleGeneGP


def cxOnePoint_random_tree_pearson(ind1: MultipleGeneGP, ind2: MultipleGeneGP, test_func):
    a = ind1.random_select()
    b = ind2.random_select()
    x, y = test_func(a), test_func(b)
    while True:
        cxOnePoint(a, b)
        z, w = test_func(a), test_func(b)
        if pearson_check(x, z) and pearson_check(y, z) and pearson_check(x, w) and pearson_check(y, w):
            break
    return ind1, ind2


def cxOnePoint_random_tree_genotype(ind1: MultipleGeneGP, ind2: MultipleGeneGP, visited_features: set):
    while True:
        a = ind1.random_select()
        b = ind2.random_select()
        cxOnePoint(a, b)
        if individual_to_tuple(a) in visited_features or individual_to_tuple(b) in visited_features:
            continue
    return ind1, ind2


def pearson_check(x, y):
    if np.abs(pearsonr(x, y)[0]) < 0.95:
        return True
    else:
        return False
