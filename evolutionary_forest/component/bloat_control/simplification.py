from typing import TYPE_CHECKING

import numpy as np
from deap.gp import PrimitiveTree

from evolutionary_forest.component.crossover.intron_based_crossover import IntronTerminal
from evolutionary_forest.component.evaluation import single_tree_evaluation
from evolutionary_forest.multigene_gp import MultipleGeneGP

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def simplify_gene(best_gene, simplification_pop):
    for o in simplification_pop:
        for gid, g, hash in zip(range(0, len(o.gene)), o.gene, o.hash_result):
            if hash in best_gene and len(best_gene[hash]) < len(g):
                # replace with a smaller gene
                o.gene[gid] = best_gene[hash]


def generate_smallest_gene_dict(population):
    best_gene = {}
    for o in population:
        o: MultipleGeneGP
        for gid, g, hash in zip(range(0, len(o.gene)), o.gene, o.hash_result):
            if hash in best_gene:
                if len(g) < len(best_gene[hash]):
                    best_gene[hash] = g
                else:
                    pass
            else:
                best_gene[hash] = g
    return best_gene


def hash_based_simplification(population, simplification_pop):
    # replace some genes with the smaller gene with equal semantics
    best_gene = generate_smallest_gene_dict(population)
    simplify_gene(best_gene, simplification_pop)


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
                    value = single_tree_evaluation(gene[tree], self.algorithm.pset, self.algorithm.X[:1])
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
