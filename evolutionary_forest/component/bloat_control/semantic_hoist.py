import copy
import random
from itertools import chain
from typing import List

import numpy as np
from deap.gp import PrimitiveTree, Terminal

from evolutionary_forest.component.crossover_mutation import hoistMutation
from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utils import gene_to_string


class SHM():
    def __init__(self,
                 algorithm: "EvolutionaryForestRegressor",
                 hoist_frequency=0,
                 iteratively_check=False,
                 key_item="String",
                 **kwargs):
        self.algorithm = algorithm
        self.hoist_frequency = hoist_frequency
        # Check again if not enough gene are hoisted
        self.iteratively_check = iteratively_check
        self.key_item = key_item

    def semantic_hoist_mutation(self, population: List[MultipleGeneGP], adaptive_hoist_probability):
        if self.algorithm.gene_num == 1:
            # Check across different individual to ensure diversity
            previous_gene = set()
            for o in sorted(population, key=lambda x: x.fitness.wvalues, reverse=True):
                if self.algorithm.bloat_control.get("diversity_enhancement", False):
                    self.hoist_mutation(o, adaptive_hoist_probability, previous_gene)
                else:
                    self.hoist_mutation(o, adaptive_hoist_probability)
        else:
            for o in population:
                self.hoist_mutation(o, adaptive_hoist_probability)

    def hoist_mutation(self, o: MultipleGeneGP, hoist_probability=None, previous_gene=None):
        hoist_frequency = self.hoist_frequency
        if hoist_frequency != 0 and self.algorithm.current_gen % hoist_frequency != 0 \
            and self.algorithm.current_gen != self.algorithm.n_gen:
            # skip hoist operation after a few iterations
            return

        if previous_gene is None:
            previous_gene = set()

        if self.algorithm.bloat_control.get("rank_by_size", False):
            order = list(enumerate(np.argsort([len(g) for g in o.gene])))
        elif self.algorithm.bloat_control.get("rank_by_size_inverse", False):
            order = list(enumerate(np.argsort([len(g) for g in o.gene])))
        elif self.algorithm.bloat_control.get("rank_by_weight", False):
            order = list(enumerate(np.argsort(-1 * o.coef)))
        else:
            order = list(enumerate(range(0, len(o.gene))))

        """
        Iteratively hoist trees, until used all trees
        """
        iteratively_check = self.iteratively_check
        key_item = self.key_item
        new_gene = []

        # rank all gene
        all_genes = [list(sorted([(k, getattr(g, 'corr', 0), gid) for k, g in enumerate(gene)],
                                 key=lambda x: (x[1], x[0]),
                                 reverse=True)) for gid, gene in enumerate(o.gene)]

        dynamic_threshold = np.mean(list(map(lambda g: g[1], chain.from_iterable(all_genes))))
        dynamic_threshold = dynamic_threshold * self.algorithm.bloat_control.get("dynamic_threshold", 0)

        dropout_probability = self.algorithm.bloat_control.get("dropout_probability", 0)
        shared_gene = self.algorithm.bloat_control.get("shared_gene", False)
        if shared_gene:
            all_genes = list(sorted(chain.from_iterable(all_genes), key=lambda x: (x[1], x[0]), reverse=True))

        while len(new_gene) < self.algorithm.gene_num:
            fail = True
            for eid, gid in order:
                gene = o.gene[gid]
                if hoist_probability is None:
                    hoist_probability = self.algorithm.bloat_control.get('hoist_probability', 1)
                hoisted = False
                if random.random() < hoist_probability:
                    """
                    Take correlation into consideration
                    """
                    if dropout_probability > 0 and random.random() < dropout_probability:
                        # Random dropout a gene
                        new_tree = self.algorithm.new_tree_generation()
                        new_gene.append(new_tree)
                        continue

                    no_check = self.algorithm.bloat_control.get("no_check", False)
                    if shared_gene:
                        gene_list = all_genes
                    else:
                        gene_list = all_genes[gid]
                    for c_index, candidate_id in enumerate(gene_list):
                        if shared_gene:
                            gene = o.gene[candidate_id[2]]
                        # get a string of a subtree
                        sub_tree_id: slice = gene.searchSubtree(candidate_id[0])
                        if key_item == 'String':
                            # String is the simplest way to avoid identical features
                            gene_str = gene_to_string(gene[sub_tree_id])
                        elif key_item == 'Correlation':
                            gene_str = gene[sub_tree_id][0].corr
                        elif key_item == 'LSH':
                            # LSH is an ideal way to avoid hoisting similar features
                            # np.unique([o.hash_id for o in chain.from_iterable(o.gene)])
                            gene_str = gene[sub_tree_id][0].hash_id
                        else:
                            raise Exception("Unexpected Key!")

                        if no_check or (gene_str not in previous_gene and
                                        getattr(gene[candidate_id[0]], 'corr', 0) > dynamic_threshold):
                            # never hoist a redundant gene or a irrelevant gene
                            previous_gene.add(gene_str)
                            new_gene.append(hoistMutation(copy.deepcopy(gene), candidate_id[0]))
                            hoisted = True
                            # any successful hoist is okay
                            fail = False
                            # all previous genes are not re-usable, thus skip them
                            if shared_gene:
                                all_genes = all_genes[c_index + 1:]
                            else:
                                all_genes[gid] = all_genes[gid][c_index + 1:]
                            break

                    if not hoisted and self.algorithm.bloat_control.get("feature_reinitialization",
                                                                        False) and not no_check:
                        if shared_gene:
                            all_genes = []
                        else:
                            all_genes[gid] = []
                        new_tree = self.algorithm.new_tree_generation()
                        new_gene.append(new_tree)
                        continue

                    if not hoisted and self.algorithm.bloat_control.get("do_nothing", False) and not no_check:
                        new_gene.append(gene)
                        continue

                    # add a zero tree
                    if not hoisted and self.algorithm.bloat_control.get("zero_tree", False) and not no_check:
                        new_gene.append(PrimitiveTree([Terminal(0, False, object)]))
                        continue

                    if not hoisted and not iteratively_check:
                        """
                        All possible operations:
                        1. Randomly initialize a new tree
                        2. Generate a zero tree
                        3. Skip checking
                        4. Iteratively check
                        """
                        raise Exception(f'There is no available operation for tree {gid}!')
                else:
                    new_gene.append(gene)

            if not iteratively_check or fail:
                break

        assert len(new_gene) >= 0
        if not iteratively_check:
            assert len(new_gene) == self.algorithm.gene_num, f'Number of Genes {len(new_gene)}'
        # may sample more trees, just taking a few is enough
        new_gene = new_gene[:self.algorithm.gene_num]
        while len(new_gene) < self.algorithm.gene_num:
            new_tree = self.algorithm.new_tree_generation()
            new_gene.append(new_tree)
        assert len(new_gene) == self.algorithm.gene_num, f'Number of Genes {len(new_gene)}'
        o.gene = new_gene
