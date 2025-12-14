import copy
import random
from functools import wraps
from typing import List

import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP


def normalize_importance(coef: np.ndarray, power: float = 1.0) -> np.ndarray:
    coef_abs = np.abs(coef)
    if coef_abs.sum() == 0:
        return np.ones_like(coef) / len(coef)
    coef_powered = np.power(coef_abs, power)
    return coef_powered / coef_powered.sum()


def revert_genes_by_importance(
    individual: MultipleGeneGP,
    original_genes: List,
    revert_probability: float,
    feature_importance_power: float = 1.0,
):
    """
    Revert genes based on importance.
    """
    if len(original_genes) != len(individual.gene) or individual.coef is None:
        return

    norm_imp = normalize_importance(individual.coef, power=feature_importance_power)

    for i, (orig_gene, imp) in enumerate(zip(original_genes, norm_imp)):
        revert_prob = revert_probability * imp
        if random.random() < revert_prob:
            individual.gene[i] = copy.deepcopy(orig_gene)


def with_revert_probability(
    revert_probability: float = 0.0,
    feature_importance_power: float = 1.0,
):
    """
    Decorator for revert probability based on feature importance.

    Args:
        revert_probability: Base probability of reverting
        feature_importance_power: Power for importance normalization
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            individuals = [arg for arg in args if isinstance(arg, MultipleGeneGP)]
            if not individuals or revert_probability == 0:
                return func(*args, **kwargs)

            # Save original genes
            original_genes_list = [copy.deepcopy(ind.gene) for ind in individuals]

            result = func(*args, **kwargs)
            result_list = result if isinstance(result, tuple) else [result]
            result_individuals = [
                ind for ind in result_list if isinstance(ind, MultipleGeneGP)
            ]

            if len(result_individuals) != len(original_genes_list):
                return result

            # Revert genes based on importance
            for ind, orig_genes in zip(result_individuals, original_genes_list):
                revert_genes_by_importance(
                    ind,
                    orig_genes,
                    revert_probability,
                    feature_importance_power,
                )

            return result

        return wrapper

    return decorator
