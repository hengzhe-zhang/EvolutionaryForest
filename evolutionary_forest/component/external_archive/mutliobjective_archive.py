import copy
from typing import List

import numpy as np
from deap.tools import selNSGA2, sortNondominated

from evolutionary_forest.multigene_gp import MultipleGeneGP


class ModelSizeArchive:
    def __init__(self, archive_size, scoring_function):
        self.archive = []
        self.archive_size = archive_size
        self.scoring_function = scoring_function

    def update(self, individuals: List[MultipleGeneGP]):
        individuals = self.archive + list(individuals)
        original_objectives = np.array(
            [individuals[i].fitness.values for i in range(len(individuals))]
        )
        if hasattr(individuals[0], "sam_loss"):
            objectives = np.array(
                [individuals[i].sam_loss for i in range(len(individuals))]
            )
        else:
            objectives = original_objectives
        size = [np.sum([len(tree) for tree in ind.gene]) for ind in individuals]
        combine_objectives = np.column_stack((objectives, size))
        assert len(combine_objectives) == len(individuals)
        for ind, values in zip(individuals, combine_objectives):
            ind.fitness.weights = (-1,) * (len(values))
            ind.fitness.values = tuple(values)
        # update archive
        pop = selNSGA2(individuals, self.archive_size)
        verbose = True
        if verbose:
            first_PF = sortNondominated(individuals, self.archive_size)[0]
            print("Archive size: ", len(first_PF))
        # restore
        for ind, values in zip(individuals, original_objectives):
            values = tuple(values)
            ind.fitness.weights = (-1,) * len(values)
            ind.fitness.values = values
        # deep copy
        self.archive = copy.deepcopy(pop)
        for ind in self.archive:
            ind.fitness.weights = (-1,) * len(ind.fitness.wvalues)

    def __iter__(self):
        return iter(self.archive)
