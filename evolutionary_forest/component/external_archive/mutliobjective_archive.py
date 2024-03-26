from typing import List

import numpy as np
from deap.tools import selNSGA2

from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utils import efficient_deepcopy


class ModelSizeArchive:
    def __init__(self, archive_size):
        self.archive = []
        self.archive_size = archive_size

    def update(self, individuals: List[MultipleGeneGP]):
        individuals = self.archive + list(individuals)
        objectives = [individuals[i].fitness.values for i in range(len(individuals))]
        size = [np.sum([len(tree) for tree in ind.gene]) for ind in individuals]
        combine_objectives = np.column_stack((objectives, size))
        assert len(combine_objectives) == len(individuals)
        for ind, values in zip(individuals, combine_objectives):
            ind.fitness.weights = (-1,) * (len(values))
            ind.fitness.values = tuple(values)
        # update archive
        pop = selNSGA2(individuals, self.archive_size)
        # restore
        for ind in individuals:
            values = tuple(ind.fitness.values[:-1])
            ind.fitness.weights = (-1,) * len(values)
            ind.fitness.values = values
        # deep copy
        self.archive = [efficient_deepcopy(ind) for ind in pop]
