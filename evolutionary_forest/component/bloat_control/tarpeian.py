import random
from typing import TYPE_CHECKING, List

import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class Tarpeian:
    def __init__(
        self, algorithm: "EvolutionaryForestRegressor", reduce_fraction=0.3, **kwargs
    ):
        self.algorithm = algorithm
        self.reduce_fraction = reduce_fraction

    def tarpeian(self, new_offspring: List[MultipleGeneGP]):
        avg_size = np.mean([len(o) for o in new_offspring])
        candidates = []
        candidates.extend(list(filter(lambda o: len(o) <= avg_size, new_offspring)))
        for o in filter(lambda o: len(o) > avg_size, new_offspring):
            # random drop some big individuals
            if random.random() > self.reduce_fraction:
                candidates.append(o)
        new_offspring = candidates
        return new_offspring
