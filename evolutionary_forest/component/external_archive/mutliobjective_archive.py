from typing import List

import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP


class ModelSizeArchive:
    def __init__(self):
        self.archive = []

    def update(self, individuals: List[MultipleGeneGP]):
        objectives = [individuals[i].fitness.values for i in range(len(individuals))]
        size = [np.sum([len(tree) for tree in ind.gene]) for ind in individuals]
        combine_objectives = np.column_stack((objectives, size))
