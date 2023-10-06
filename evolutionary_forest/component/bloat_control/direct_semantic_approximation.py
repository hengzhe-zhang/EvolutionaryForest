import random

import numpy as np
from deap.gp import Terminal

from evolutionary_forest.component.evaluation import single_tree_evaluation
from evolutionary_forest.multigene_gp import quick_fill, MultipleGeneGP


class DSA():
    def __init__(self,
                 algorithm: "EvolutionaryForestRegressor",
                 **kwargs):
        self.algorithm = algorithm

    def subtree_semantic_approximation(self, *population: MultipleGeneGP):
        toolbox = self.algorithm.toolbox
        mean_size = np.mean([len(p.gene) for p in population])
        for p in population:
            for g in p.gene:
                if len(g) > mean_size:
                    index = random.randrange(len(g))
                    slice_ = g.searchSubtree(index)
                    if slice_.stop - slice_.start <= 3:
                        continue
                    type_ = g[index].ret
                    old_y = single_tree_evaluation(g[slice_], self.algorithm.pset, self.algorithm.X)
                    old_y = quick_fill([old_y], self.algorithm.y)[0]
                    new_ind = None
                    # 1000 trails
                    for _ in range(0, 1000):
                        new_ind = toolbox.expr_mut(pset=self.algorithm.pset, type_=type_)
                        if len(new_ind) < slice_.stop - slice_.start - 2:
                            break
                    if new_ind is not None:
                        new_y = single_tree_evaluation(new_ind, self.algorithm.pset, self.algorithm.X)
                        new_y = quick_fill([new_y], self.algorithm.y)[0]
                        theta = np.dot(new_y, old_y) / np.dot(new_y, new_y)
                        new_ind.insert(0, self.algorithm.pset.mapping["Mul"])
                        new_ind.insert(1, Terminal(theta, False, object))
                        g[slice_] = new_ind
        return population
