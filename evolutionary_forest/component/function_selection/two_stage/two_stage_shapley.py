from typing import Callable, List

import numpy as np
import shap
from deap.gp import PrimitiveSet
from sklearn.inspection import permutation_importance

from evolutionary_forest.model.gp_tree_wrapper import GPWrapper
from evolutionary_forest.multigene_gp import MultipleGeneGP


def remove_functions(elements_to_remove: List, pset: PrimitiveSet):
    for element_name in elements_to_remove:
        # Check and remove from pset.primitives
        for return_type, primitives in list(pset.primitives.items()):
            for p in primitives:
                if p.name == element_name:
                    pset.primitives[return_type].remove(p)
                    break

        # Check and remove from pset.terminals
        for return_type, terminals in list(pset.terminals.items()):
            for t in terminals:
                if t.name == element_name:
                    pset.terminals[return_type].remove(t)
                    break


def two_stage_feature_selection(
    pop: List[MultipleGeneGP], pset: PrimitiveSet, feature_generation: Callable, X, y
):
    importance_dict = {f"ARG{i}": 0 for i in range(X.shape[1])}

    for ind in sorted(pop, key=lambda x: x.fitness.wvalues[0], reverse=True)[:30]:
        importance = permutation_feature_importance(ind, feature_generation, X, y)
        for i, value in enumerate(importance):
            key = f"ARG{i}"
            if key in importance_dict:
                importance_dict[key] += value
            else:
                importance_dict[key] = value

    top_10 = sorted(importance_dict, key=importance_dict.get, reverse=True)[:10]
    remaining_features = set(importance_dict.keys()) - set(top_10)
    remove_functions(list(remaining_features), pset)


def shapley_feature_selection(
    ind: MultipleGeneGP, feature_generation: Callable, X, y
) -> np.ndarray:
    gp_model = GPWrapper(ind, feature_generation, None, None)
    model = lambda data: gp_model.predict(data)
    data = X
    explainer = shap.SamplingExplainer(model=model, data=X)

    # Calculate Shapley values
    shap_values = explainer.shap_values(data, silent=True)
    average_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
    return average_abs_shap_values


def permutation_feature_importance(
    ind: MultipleGeneGP, feature_generation: Callable, X, y
):
    gp_model = GPWrapper(ind, feature_generation, None, None)
    result = permutation_importance(gp_model, X, y, scoring="r2", random_state=0)
    feature_importance = result.importances_mean
    return feature_importance
