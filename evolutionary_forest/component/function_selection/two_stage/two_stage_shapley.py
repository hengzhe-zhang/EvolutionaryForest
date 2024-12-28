from collections import Counter
from typing import Callable, List

import numpy as np
import shap
from deap import gp
from deap.gp import PrimitiveSet, Terminal
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
    pop: List[MultipleGeneGP],
    pset: PrimitiveSet,
    feature_generation: Callable,
    X,
    y,
    mode,
):
    importance_dict = {f"ARG{i}": 0 for i in range(X.shape[1])}

    for ind in sorted(pop, key=lambda x: x.fitness.wvalues[0], reverse=True)[:30]:
        if mode == "Permutation":
            importance = permutation_feature_importance(ind, feature_generation, X, y)
        elif mode == "Shapley":
            importance = shapley_feature_selection(
                ind, feature_generation, X, y, nsamples=5
            )
        elif mode == "Frequency":
            importance = frequency_feature_selection(ind, X)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        for i, value in enumerate(importance):
            key = f"ARG{i}"
            if key in importance_dict:
                importance_dict[key] += value
            else:
                importance_dict[key] = value

    # Determine the threshold k using log2 and round to the nearest integer
    k = int(np.log2(X.shape[1]).round())
    top_k = sorted(importance_dict, key=importance_dict.get, reverse=True)[:k]
    remaining_features = set(importance_dict.keys()) - set(top_k)
    remove_functions(list(remaining_features), pset)


def frequency_feature_selection(ind: MultipleGeneGP, X) -> np.ndarray:
    # Dictionary to count occurrences of each terminal (feature)
    importance = Counter()

    # Iterate over each tree in the individual
    for tree in ind.gene:
        # Ensure the tree is a PrimitiveTree from DEAP
        tree: gp.PrimitiveTree

        # Iterate over nodes in the tree
        for node in tree:
            # Check if the node is a terminal (feature)
            if isinstance(node, Terminal):
                importance[node.name] += 1

    frequencies = np.array([importance.get(f"ARG{i}", 0) for i in range(X.shape[1])])

    return frequencies


def shapley_feature_selection(
    ind: MultipleGeneGP, feature_generation: Callable, X, y, nsamples=10
) -> np.ndarray:
    # Dictionary to count occurrences of each terminal (feature)
    importance = Counter()

    # Iterate over each tree in the individual
    for tree in ind.gene:
        for node in tree:
            # Check if the node is a terminal (feature)
            if isinstance(node, Terminal):  # Ensure Terminal is defined
                importance[node.name] += 1

    # Initialize Shapley values array
    default_shap = np.zeros(X.shape[1])

    # Identify used features
    used_features = [f for f in range(X.shape[1]) if f"ARG{f}" in importance]
    X_used = X[:, used_features]

    # Define the model as a function to maintain the feature positions
    def model(data):
        # Fill the data to match the original feature size, maintaining the positions
        full_data = np.zeros((data.shape[0], X.shape[1]))
        full_data[:, used_features] = data
        gp_model = GPWrapper(ind, feature_generation, None, None)
        return gp_model.predict(full_data)

    # Background data for SHAP
    background_data = np.mean(X_used, axis=0).reshape(1, -1)

    # Initialize SHAP explainer
    explainer = shap.SamplingExplainer(
        model=model, data=background_data, nsamples=nsamples
    )

    # Calculate Shapley values
    shap_values = explainer.shap_values(
        X_used[np.random.choice(X_used.shape[0], 10)], silent=True
    )

    # Compute average absolute Shapley values
    average_abs_shap_values = np.zeros_like(default_shap)
    average_abs_shap_values[used_features] = np.mean(np.abs(shap_values), axis=0)

    return average_abs_shap_values


def permutation_feature_importance(
    ind: MultipleGeneGP, feature_generation: Callable, X, y
):
    gp_model = GPWrapper(ind, feature_generation, None, None)
    result = permutation_importance(gp_model, X, y, scoring="r2", random_state=0)
    feature_importance = result.importances_mean
    return feature_importance
