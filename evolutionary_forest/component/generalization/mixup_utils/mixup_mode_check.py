import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score

from evolutionary_forest.component.generalization.pac_bayesian import (
    PACBayesianConfiguration,
)


def mixup_mode_check_and_change(X_data, y_data, pac_bayesian: PACBayesianConfiguration):
    mixup_mode = pac_bayesian.mixup_mode
    adaptive_knee_point_metric = pac_bayesian.adaptive_knee_point_metric
    if isinstance(mixup_mode, str):
        score = np.mean(
            cross_val_score(ExtraTreesRegressor(), X_data, y_data, cv=5, scoring="r2")
        )
        if mixup_mode == "Adaptive-ET":
            if score < 0.5:
                mixup_mode = ""
            else:
                mixup_mode = "RBF,ET,0.05"
        if adaptive_knee_point_metric == "Adaptive":
            if score <= 0.75:
                adaptive_knee_point_metric = 10
            else:
                adaptive_knee_point_metric = 1
    pac_bayesian.mixup_mode = mixup_mode
    pac_bayesian.adaptive_knee_point_metric = adaptive_knee_point_metric
