import copy
import random
from functools import wraps
from typing import List

import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP


def save_original_genes(*individuals: MultipleGeneGP) -> List[List]:
    return [copy.deepcopy(ind.gene) for ind in individuals]


def normalize_importance(coef: np.ndarray) -> np.ndarray:
    coef_abs = np.abs(coef)
    if coef_abs.sum() == 0:
        return np.ones_like(coef) / len(coef)
    return coef_abs / coef_abs.sum()


def revert_genes_by_importance(
    individual: MultipleGeneGP, original_genes: List, revert_probability: float
):
    if len(original_genes) != len(individual.gene):
        return

    if not hasattr(individual, "coef") or individual.coef is None:
        return

    normalized_importance = normalize_importance(individual.coef)
    for i, (orig_gene, curr_gene) in enumerate(zip(original_genes, individual.gene)):
        if i < len(normalized_importance):
            gene_revert_prob = revert_probability * normalized_importance[i]
            if random.random() < gene_revert_prob:
                individual.gene[i] = copy.deepcopy(orig_gene)


def with_revert_probability(revert_probability: float = 0.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            individuals = [arg for arg in args if isinstance(arg, MultipleGeneGP)]
            if not individuals:
                return func(*args, **kwargs)

            original_genes = save_original_genes(*individuals)
            result = func(*args, **kwargs)

            if revert_probability > 0:
                result_individuals = result if isinstance(result, tuple) else [result]
                result_individuals = [
                    ind for ind in result_individuals if isinstance(ind, MultipleGeneGP)
                ]
                if len(result_individuals) == len(original_genes):
                    for ind, orig_genes in zip(result_individuals, original_genes):
                        revert_genes_by_importance(ind, orig_genes, revert_probability)

            return result

        return wrapper

    return decorator
