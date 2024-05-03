from typing import List

import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP


def get_diversity_matrix(all_ind: List[MultipleGeneGP]):
    """
    Get the diversity matrix of the population
    :param all_ind: List of individuals
    :return: Diversity matrix
    """
    inds = []
    for p in all_ind:
        inds.append(p.predicted_values.flatten())
    inds = np.array(inds)
    return inds
