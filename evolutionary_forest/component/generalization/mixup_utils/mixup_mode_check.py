import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score


def mixup_mode_check(X_data, y_data, mixup_mode):
    if isinstance(mixup_mode, str):
        score = np.mean(
            cross_val_score(ExtraTreesRegressor(), X_data, y_data, cv=5, scoring="r2")
        )
        if mixup_mode == "Adaptive-ET":
            if score < 0.5:
                mixup_mode = ""
            else:
                mixup_mode = "RBF,ET,0.05"
    return mixup_mode
