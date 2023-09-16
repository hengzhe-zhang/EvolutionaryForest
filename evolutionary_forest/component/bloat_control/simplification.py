from typing import TYPE_CHECKING

import numpy as np
from deap.gp import PrimitiveTree

from evolutionary_forest.component.crossover.intron_based_crossover import IntronTerminal
from evolutionary_forest.component.evaluation import quick_evaluate

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class Simplification():
    def __init__(self, algorithm: "EvolutionaryForestRegressor",
                 constant_prune=True,
                 equal_prune=True,
                 **kwargs):
        self.algorithm = algorithm
        self.constant_prune = constant_prune
        self.equal_prune = equal_prune

    def gene_prune(self, gene: PrimitiveTree):
        if self.algorithm.bloat_control is None or (not self.algorithm.bloat_control.get('hoist_mutation', False)):
            return
        constant_prune = self.algorithm.bloat_control.get('constant_prune', True)
        if constant_prune:
            # constant prune
            gid = 0
            while gid < len(gene):
                if getattr(gene[gid], 'corr', None) == -1:
                    tree = gene.searchSubtree(gid)
                    value = quick_evaluate(gene[tree], self.algorithm.pset, self.algorithm.X[:1])
                    if isinstance(value, np.ndarray):
                        value = float(value.flatten()[0])
                    c = IntronTerminal(value, False, object)
                    gene[tree] = [c]
                gid += 1
        equal_prune = self.algorithm.bloat_control.get('equal_prune', True)
        if equal_prune:
            gid = 0
            while gid < len(gene):
                equal_subtree = getattr(gene[gid], 'equal_subtree', -1)
                if equal_subtree >= 0:
                    tree = gene.searchSubtree(gid)
                    subtree = gid + 1
                    subtree_id = 0
                    while subtree_id < equal_subtree:
                        subtree = gene.searchSubtree(subtree).stop
                        subtree_id += 1
                    gene[tree] = gene[gene.searchSubtree(subtree)]
                gid += 1
