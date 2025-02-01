import random
from collections import defaultdict

from deap.gp import __type__

from evolutionary_forest.component.configuration import (
    CrossoverConfiguration,
)


def normalize_by_to_one_eps(weights, reverse=False):
    # Clip weights to avoid zeros
    weights = [w + 1e-10 for w in weights]

    if reverse:
        weights = [1 / (w + 1e-10) for w in weights]  # Inverse for lower weights

    total = sum(weights)

    # Normalize weights to probabilities
    probabilities = [w / total for w in weights]

    return random.choices(range(len(weights)), weights=probabilities, k=1)[0]


def normalize_by_to_one(weights, reverse=False):
    # Clip weights to avoid zeros
    weights = [w + 1 for w in weights]

    if reverse:
        weights = [1 / (w + 1) for w in weights]  # Inverse for lower weights

    total = sum(weights)

    # Normalize weights to probabilities
    probabilities = [w / total for w in weights]

    return random.choices(range(len(weights)), weights=probabilities, k=1)[0]


def cxOnePointWeighted(ind1, ind2, configuration: CrossoverConfiguration):
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    # Not STGP optimization
    assert ind1.root.ret == __type__
    types1[__type__] = list(range(0, len(ind1)))
    types2[__type__] = list(range(0, len(ind2)))
    common_types = [__type__]

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        # Weighted selection of "good" and "bad" subtrees
        index1_good = normalize_by_to_one(ind1.subtree_weights)
        index2_good = normalize_by_to_one(ind2.subtree_weights)
        assert index1_good < len(ind1), f"{index1_good} >= {len(ind1)}"
        assert index2_good < len(ind2), f"{index2_good} >= {len(ind2)}"
        index1_bad = normalize_by_to_one(ind1.subtree_weights, reverse=True)
        index2_bad = normalize_by_to_one(ind2.subtree_weights, reverse=True)

        # Find slices of the selected subtrees
        slice1_good = ind1.searchSubtree(index1_good)
        slice2_good = ind2.searchSubtree(index2_good)
        slice1_bad = ind1.searchSubtree(index1_bad)
        slice2_bad = ind2.searchSubtree(index2_bad)

        # Perform weighted crossover: Replace "bad" subtree with "good" subtree
        ind1[slice1_bad], ind2[slice2_bad] = ind2[slice2_good], ind1[slice1_good]

    return ind1, ind2


def cxOnePointWeightedEPS(ind1, ind2, configuration: CrossoverConfiguration):
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    # Not STGP optimization
    assert ind1.root.ret == __type__
    types1[__type__] = list(range(0, len(ind1)))
    types2[__type__] = list(range(0, len(ind2)))
    common_types = [__type__]

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        # Weighted selection of "good" and "bad" subtrees
        index1_good = normalize_by_to_one_eps(ind1.subtree_weights)
        index2_good = normalize_by_to_one_eps(ind2.subtree_weights)
        assert index1_good < len(ind1), f"{index1_good} >= {len(ind1)}"
        assert index2_good < len(ind2), f"{index2_good} >= {len(ind2)}"
        index1_bad = normalize_by_to_one_eps(ind1.subtree_weights, reverse=True)
        index2_bad = normalize_by_to_one_eps(ind2.subtree_weights, reverse=True)

        # Find slices of the selected subtrees
        slice1_good = ind1.searchSubtree(index1_good)
        slice2_good = ind2.searchSubtree(index2_good)
        slice1_bad = ind1.searchSubtree(index1_bad)
        slice2_bad = ind2.searchSubtree(index2_bad)

        # Perform weighted crossover: Replace "bad" subtree with "good" subtree
        ind1[slice1_bad], ind2[slice2_bad] = ind2[slice2_good], ind1[slice1_good]

    return ind1, ind2


def cxOnePointWeightedRandom(ind1, ind2, configuration: CrossoverConfiguration):
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    # Not STGP optimization
    assert ind1.root.ret == __type__
    types1[__type__] = list(range(0, len(ind1)))
    types2[__type__] = list(range(0, len(ind2)))
    common_types = [__type__]

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        # Weighted selection of "good" and "bad" subtrees
        index1_good = normalize_by_to_one(ind1.subtree_weights)
        index2_good = normalize_by_to_one(ind2.subtree_weights)
        assert index1_good < len(ind1), f"{index1_good} >= {len(ind1)}"
        assert index2_good < len(ind2), f"{index2_good} >= {len(ind2)}"
        index1_bad = random.choice(types1[__type__])
        index2_bad = random.choice(types2[__type__])

        # Find slices of the selected subtrees
        slice1_good = ind1.searchSubtree(index1_good)
        slice2_good = ind2.searchSubtree(index2_good)
        slice1_bad = ind1.searchSubtree(index1_bad)
        slice2_bad = ind2.searchSubtree(index2_bad)

        # Perform weighted crossover: Replace "bad" subtree with "good" subtree
        ind1[slice1_bad], ind2[slice2_bad] = ind2[slice2_good], ind1[slice1_good]

    return ind1, ind2
