import copy
import random
from typing import TYPE_CHECKING, List

from deap.gp import Primitive

from evolutionary_forest.component.crossover_mutation import hoistMutation, hoistMutationWithTerminal
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
                # avoid to prune a trivial tree with only one node, as it cannot reduce the tree size
                primitive_nodes = list(
                    filter(lambda x: x != 0 and isinstance(gene[x], Primitive), range(0, len(gene))))
                if len(primitive_nodes) == 0:
                    continue
                if best:
                    best_id = max([(k, getattr(gene[k], 'corr', 0)) for k in primitive_nodes],
                                  key=lambda x: (x[1], x[0]))[0]
                else:
                    best_id = random.choice(primitive_nodes)
                # small tree
                hoistMutation(new_gene, best_id)
                # remove small tree from original tree
                hoistMutationWithTerminal(gene, best_id, self.algorithm.pset,
                                          self.algorithm.estimation_of_distribution.terminal_prob)
            new_population.append(o)
            new_population.append(new_o)
        offspring = new_population
        return offspring
