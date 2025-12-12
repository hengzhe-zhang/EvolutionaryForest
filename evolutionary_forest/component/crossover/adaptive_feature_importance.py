import copy
import random
from functools import wraps
from typing import List, Optional

import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP


def save_original_genes(*individuals: MultipleGeneGP) -> List[List]:
    return [copy.deepcopy(ind.gene) for ind in individuals]


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
    if len(original_genes) != len(individual.gene) or individual.coef is None:
        return

    normalized_importance = normalize_importance(
        individual.coef, power=feature_importance_power
    )
    for i, (orig_gene, norm_imp) in enumerate(
        zip(original_genes, normalized_importance)
    ):
        if random.random() < revert_probability * norm_imp:
            individual.gene[i] = copy.deepcopy(orig_gene)


def revert_after_evaluation_by_importance(
    individual: MultipleGeneGP,
    revert_probability: float,
    feature_importance_power: float = 1.0,
):
    # Only process individuals that have the revert attributes set
    # (i.e., those that went through crossover/mutation with the decorator)
    if not hasattr(individual, "parent_genes_for_revert") or not hasattr(
        individual, "parent_coef_for_revert"
    ):
        return

    parent_genes = individual.parent_genes_for_revert
    parent_coef = individual.parent_coef_for_revert

    assert individual.coef is not None
    assert len(parent_genes) == len(individual.gene)
    assert len(parent_coef) == len(individual.coef)

    normalized_importance = normalize_importance(
        individual.coef, power=feature_importance_power
    )

    for i, (curr_coef, parent_c, norm_imp, parent_gene) in enumerate(
        zip(individual.coef, parent_coef, normalized_importance, parent_genes)
    ):
        if abs(curr_coef) < abs(parent_c):
            if random.random() < revert_probability * norm_imp:
                individual.gene[i] = copy.deepcopy(parent_gene)

    delattr(individual, "parent_genes_for_revert")
    delattr(individual, "parent_coef_for_revert")


def with_revert_probability(
    revert_probability: float = 0.0,
    feature_importance_power: float = 1.0,
    postpone_to_after_evaluation: bool = False,
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            individuals = [arg for arg in args if isinstance(arg, MultipleGeneGP)]
            if not individuals:
                return func(*args, **kwargs)

            original_genes = save_original_genes(*individuals)
            original_coefs = (
                [
                    copy.deepcopy(ind.coef) if ind.coef is not None else None
                    for ind in individuals
                ]
                if postpone_to_after_evaluation and revert_probability > 0
                else []
            )

            result = func(*args, **kwargs)

            if revert_probability > 0:
                result_list = result if isinstance(result, tuple) else [result]
                result_individuals = [
                    ind for ind in result_list if isinstance(ind, MultipleGeneGP)
                ]
                if len(result_individuals) == len(original_genes):
                    if postpone_to_after_evaluation:
                        for ind, orig_genes, orig_coef in zip(
                            result_individuals, original_genes, original_coefs
                        ):
                            ind.parent_genes_for_revert = copy.deepcopy(orig_genes)
                            if orig_coef is not None:
                                ind.parent_coef_for_revert = orig_coef
                    else:
                        for ind, orig_genes in zip(result_individuals, original_genes):
                            revert_genes_by_importance(
                                ind,
                                orig_genes,
                                revert_probability,
                                feature_importance_power,
                            )

            return result

        return wrapper

    return decorator
