import copy
import random
from inspect import isclass
from typing import TYPE_CHECKING, List

import numpy as np
from deap.gp import Primitive

from evolutionary_forest.component.crossover_mutation import hoistMutation
from evolutionary_forest.multigene_gp import MultipleGeneGP

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class PAP():
    def __init__(self, algorithm: "EvolutionaryForestRegressor"):
        self.algorithm = algorithm

    def prune_and_plant(self, offspring: List[MultipleGeneGP], best=False):
        new_population = []
        for o in offspring:
            new_o = copy.deepcopy(o)
            for id, gene, new_gene in zip(range(0, len(o.gene)), o.gene, new_o.gene):
                # avoid pruning a trivial tree with only one node, as it cannot reduce the tree size
                primitive_nodes = list(filter(lambda x: x != 0 and isinstance(gene[x], Primitive),
                                              range(len(gene))))
                if len(primitive_nodes) == 0:
                    continue
                if best:
                    best_id = max(((k, getattr(gene[k], 'corr', 0)) for k in primitive_nodes),
                                  key=lambda x: (x[1], x[0]))[0]
                else:
                    best_id = random.choice(primitive_nodes)
                # small tree
                hoistMutation(new_gene, best_id)
                # remove a small tree from an original tree
                hoistMutationWithTerminal(gene, best_id, self.algorithm.pset,
                                          self.algorithm.estimation_of_distribution.terminal_prob)
            new_population.extend((o, new_o))
        offspring = new_population
        return offspring


def hoistMutationWithTerminal(ind, best_index, pset, terminal_probs=None):
    """
    Bloat control: Replace one subtree with a constant
    """
    sub_slice = ind.searchSubtree(best_index)
    if terminal_probs is None:
        # Terminal without a probability
        terminal_node = random.choice(pset.terminals[pset.ret])
    else:
        terminal_probs = terminal_probs / terminal_probs.sum()
        terminal_node = np.random.choice(pset.terminals[pset.ret], p=terminal_probs)
    if isclass(terminal_node):
        # For random constants
        terminal_node = terminal_node()
    ind[sub_slice] = [terminal_node]
    return ind
