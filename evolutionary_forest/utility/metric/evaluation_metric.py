import numpy as np
from sklearn.metrics import r2_score


def safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    # Replace NaN and inf values in y_pred with 0 using nan_to_num
    y_pred_clean = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute and return the R² score on the cleaned y_pred
    return r2_score(y_true, y_pred_clean)
