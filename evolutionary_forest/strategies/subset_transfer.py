from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor

from evolutionary_forest.multigene_gp import MultipleGeneGP


def train_final_model_on_full_batch(
    model: MultipleGeneGP,
    ef: "EvolutionaryForestRegressor",
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    constructed_features = ef.feature_construction(X_train, model)
    model.pipe.fit(constructed_features, y_train.flatten())
