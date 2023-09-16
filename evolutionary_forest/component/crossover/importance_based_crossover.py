from deap.gp import cxOnePoint

from evolutionary_forest.multigene_gp import MultipleGeneGP


def cxOnePoint_multi_tree_weighted(ind1: MultipleGeneGP, ind2: MultipleGeneGP):
    # Potential issue: This operator may overly cross important features and ignore useless features
    cxOnePoint(ind1.weighted_selection(), ind2.weighted_selection())
    return ind1, ind2
