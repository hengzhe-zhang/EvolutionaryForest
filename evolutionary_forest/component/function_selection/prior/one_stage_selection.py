from typing import TYPE_CHECKING

import numpy as np
import shap
import xgboost as xgb
from deap.gp import PrimitiveSet
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from evolutionary_forest.component.function_selection.two_stage.two_stage_shapley import (
    remove_functions,
)

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def get_terminal_probability(self: "EvolutionaryForestRegressor"):
    """
    Using feature importance at initialization
    """
    # Get a probability distribution based on the importance of original features in a random forest
    if isinstance(self, ClassifierMixin):
        r = RandomForestClassifier(n_estimators=5)
    else:
        r = RandomForestRegressor(n_estimators=5)
    r.fit(self.X, self.y)
    terminal_prob = np.append(r.feature_importances_, 0.1)
    terminal_prob = terminal_prob / np.sum(terminal_prob)
    return terminal_prob


def prior_feature_selection(pset: PrimitiveSet, X, y, mode: str):
    importance_dict = {f"ARG{i}": 0 for i in range(X.shape[1])}

    if mode == "RandomForestPermutation":
        model = RandomForestRegressor(random_state=0)
        model.fit(X, y)
        perm_importance = permutation_importance(model, X, y, random_state=0)
        for i, importance in enumerate(perm_importance.importances_mean):
            importance_dict[f"ARG{i}"] = importance

    elif mode == "RandomForestShapley":
        model = RandomForestRegressor(random_state=0)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        feature_importances = np.abs(shap_values).mean(axis=0)
        for i, importance in enumerate(feature_importances):
            importance_dict[f"ARG{i}"] = importance

    elif mode == "XGBPermutation":
        model = xgb.XGBRegressor(random_state=0)
        model.fit(X, y)
        perm_importance = permutation_importance(model, X, y, random_state=0)
        for i, importance in enumerate(perm_importance.importances_mean):
            importance_dict[f"ARG{i}"] = importance

    elif mode == "XGBShapley":
        model = xgb.XGBRegressor(random_state=0)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        feature_importances = np.abs(shap_values).mean(axis=0)
        for i, importance in enumerate(feature_importances):
            importance_dict[f"ARG{i}"] = importance

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Determine the threshold k using log2 and round to the nearest integer
    k = int(np.log2(X.shape[1]).round())

    # Select top k features based on importance
    top_k = sorted(importance_dict, key=importance_dict.get, reverse=True)[:k]

    # Identify remaining features
    remaining_features = set(importance_dict.keys()) - set(top_k)

    remove_functions(list(remaining_features), pset)
