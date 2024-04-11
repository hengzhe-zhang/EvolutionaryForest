import copy
import random
from collections import defaultdict
from functools import partial
from operator import eq, lt
from typing import List

from deap.gp import __type__, PrimitiveTree, Primitive

from evolutionary_forest.component.configuration import (
    CrossoverConfiguration,
    MutationConfiguration,
)
from evolutionary_forest.component.shared_type import Parameter, LearnedParameter


def individual_combination(offspring, toolbox, pset, limitation_check):
    @limitation_check
    def combination(*population):
        assert len(population) == 2
        offspring = [toolbox.clone(ind) for ind in population]
        for gene1, gene2 in zip(offspring[0].gene, offspring[1].gene):
            gene1[0 : len(gene1)] = tree_combination(gene1, gene2, pset)
        return (offspring[0],)

    return combination(*offspring)


def tree_combination(ind1, ind2, pset):
    # combine two trees into a single tree
    type_ = pset.ret
    primitives = list(filter(lambda x: x.arity == 2, pset.primitives[type_]))

    new_head1 = random.choice(primitives)
    new_ind = copy.deepcopy(ind1)
    new_ind.insert(0, new_head1)
    new_ind.extend(ind2)
    return new_ind


def mutUniformSizeSafe(
    individual: PrimitiveTree, expr, pset, configuration: MutationConfiguration
):
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    tree_height = configuration.max_height - get_height_list(individual)[index]
    # generate a smaller tree
    individual[slice_] = expr(pset=pset, type_=type_, min_=0, max_=tree_height)
    return (individual,)


def get_children_list(prefix_list):
    stack = []
    children_list = defaultdict(list)
    for i, item in reversed(list(enumerate(prefix_list))):
        if item.arity == 0:
            stack.append(i)
        else:
            children = []
            for _ in range(item.arity):
                children.append(stack.pop())
            children.reverse()
            children_list[i] = children
            stack.append(i)
    return children_list


def inverse_height(node_idx, children_list, heights):
    if heights[node_idx] != -1:
        return heights[node_idx]
    max_height = -1
    for child in children_list[node_idx]:
        child_height = inverse_height(child, children_list, heights)
        max_height = max(max_height, child_height)
    heights[node_idx] = max_height + 1
    return heights[node_idx]


def calc_heights(heights, nodes):
    """
    Here is a list in which each sub-list represents the child nodes of each node.
    Now, the function aims to get a list of the heights of all nodes.
    The height of the root node is 0, and so on.
    """

    def dfs(node, height):
        heights[node] = height
        for child in nodes[node]:
            dfs(child, height + 1)

    dfs(0, 0)
    return heights


def get_inverse_height_list(prefix_list: List[Primitive]):
    children_list = get_children_list(prefix_list)
    heights = [-1 for i in range(len(prefix_list))]
    inverse_height(0, children_list, heights)
    return heights


def get_height_list(prefix_list: List[Primitive]):
    children_list = get_children_list(prefix_list)
    heights = [0] * len(prefix_list)
    return calc_heights(heights, children_list)


def cxOnePointSizeSafe(ind1, ind2, configuration: CrossoverConfiguration):
    """
    The condition for the effectiveness of this operator is that there are a lot of failures after the variation process.
    It is necessary to check the frequency of failure operators.
    """
    # A size-safe crossover, meaning that this operator will never disobey the max height constraint
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    # Do not support STGP optimization
    if configuration.leaf_biased:
        leaf_biased_indices(ind1, ind2, types1, types2)
        if len(types1[__type__]) == 0 or len(types2[__type__]) == 0:
            # no primitives
            return ind1, ind2
    else:
        types1[__type__] = list(range(0, len(ind1)))
        types2[__type__] = list(range(0, len(ind2)))

    common_types = [__type__]

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        # get a subtree from tree 1
        c1_index1 = random.choice(types1[type_])
        # If max height is 2 and the current height is 0, then only allow the tree height no more than 2
        index1_height = configuration.max_height - get_height_list(ind1)[c1_index1]
        # This operator ensures the success of crossover
        indices = [
            i
            for i, height in enumerate(get_inverse_height_list(ind2))
            if height <= index1_height
        ]
        if configuration.leaf_biased:
            # get index from tree 2
            c1_index2 = random.choice(list(set(indices) & set(types2[__type__])))
        else:
            c1_index2 = random.choice(indices)

        # get a subtree from tree 2
        c2_index2 = random.choice(types2[type_])
        index2_height = configuration.max_height - get_height_list(ind2)[c2_index2]
        # This operator ensures the success of crossover
        indices = [
            i
            for i, height in enumerate(get_inverse_height_list(ind1))
            if height <= index2_height
        ]
        if configuration.leaf_biased:
            # get index from tree 1
            c2_index1 = random.choice(list(set(indices) & set(types1[__type__])))
        else:
            c2_index1 = random.choice(indices)

        # pivot tree in first crossover
        c1_slice1 = ind1.searchSubtree(c1_index1)
        # smaller GP tree in first crossover
        c1_slice2 = ind2.searchSubtree(c1_index2)
        # smaller GP tree in second crossover
        c2_slice1 = ind1.searchSubtree(c2_index1)
        # pivot tree in second crossover
        c2_slice2 = ind2.searchSubtree(c2_index2)
        ind1[c1_slice1], ind2[c2_slice2] = ind2[c1_slice2], ind1[c2_slice1]

    return ind1, ind2


def leaf_biased_indices(ind1, ind2, types1, types2):
    # Determine whether to keep terminals or primitives for each individual
    termpb = 0.1
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op = terminal_op if random.random() < termpb else primitive_op
    for idx, node in enumerate(ind1):
        if arity_op(node.arity):
            types1[__type__].append(idx)
    for idx, node in enumerate(ind2):
        if arity_op(node.arity):
            types2[__type__].append(idx)


def cxOnePointWithRoot(ind1, ind2, configuration: CrossoverConfiguration):
    # A very special crossover operator designed for MGP
    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        if configuration.leaf_biased:
            leaf_biased_indices(ind1, ind2, types1, types2)
            if len(types1[__type__]) == 0 or len(types2[__type__]) == 0:
                # no primitives
                return ind1, ind2
        else:
            # Not STGP optimization
            types1[__type__] = list(range(0, len(ind1)))
            types2[__type__] = list(range(0, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[0:], 0):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[0:], 0):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))
        for type in [LearnedParameter, Parameter]:
            if type in common_types:
                common_types.remove(type)

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        # # Maybe only allowing to exchange two features is good
        # if index1 == 0:
        #     index2 = 0
        # else:
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def hoistMutation(ind, best_index):
    sub_slice = ind.searchSubtree(best_index)
    ind[0 : len(ind)] = ind[sub_slice]
    return ind
