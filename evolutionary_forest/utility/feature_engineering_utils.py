from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from evolutionary_forest.component.evaluation import single_tree_evaluation
from evolutionary_forest.utils import code_generation

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def combine_features(
    regr: "EvolutionaryForestRegressor", X_input, feature_list, only_new_features=False
):
    if isinstance(X_input, pd.DataFrame):
        X = X_input.to_numpy()
    else:
        X = X_input

    gp_tree_dict = mapping_str_tree(regr)

    # if regr.remove_constant_features:
    #     X = X[:, regr.columns_without_constants]
    if only_new_features:
        data = []
    else:
        data = [X]
    for f in feature_list:
        features = single_tree_evaluation(gp_tree_dict[f], regr.pset, X)
        if isinstance(features, np.ndarray) and len(features) == len(X):
            data.append(features.reshape(-1, 1))
    transformed_features = np.concatenate(data, axis=1)
    # Fix outliers (In extreme cases, some functions will produce unexpected results)
    transformed_features = np.nan_to_num(transformed_features, posinf=0, neginf=0)
    if isinstance(X_input, pd.DataFrame):
        if only_new_features:
            transformed_features = pd.DataFrame(
                transformed_features, columns=[f.split(":")[1] for f in feature_list]
            )
        else:
            transformed_features = pd.DataFrame(
                transformed_features,
                columns=list(X_input.columns) + [f.split(":")[1] for f in feature_list],
            )
    return transformed_features


def mapping_str_tree(regr):
    gp_tree_dict = {}
    for ind in regr.hof:
        for tree in ind.gene:
            gp_tree_dict[code_generation(regr, tree)] = tree
    return gp_tree_dict
