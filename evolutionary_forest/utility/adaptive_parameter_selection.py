import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score


def compute_adaptive_gene_num_cv(x, y, random_state=None, threshold=0.8, cv=5):
    """
    Compute adaptive gene_num based on CV ExtraTree R2 score.
    Returns 10 if CV R2 < threshold, else np.clip(x.shape[1], 10, 20).
    """
    et_model = ExtraTreesRegressor(random_state=random_state)
    cv_scores = cross_val_score(et_model, x, y, cv=cv, scoring="r2")
    cv_r2 = np.mean(cv_scores)

    if cv_r2 < threshold:
        return 10
    else:
        return int(np.clip(x.shape[1], 10, 20))


def compute_adaptive_gene_num_data_aware(x, y, random_state=None, threshold=0.8, cv=5):
    """
    Compute adaptive gene_num based on data characteristics (sample size, feature count, and model performance).
    - If number of samples < 200: return 5
    - Else if extra tree CV R2 > threshold AND number of features > 20: return 20
    - Otherwise: return 10
    """
    # First check: if number of samples < 200, return 5
    if x.shape[0] < 200:
        return 5
    
    # Compute CV R2 score
    et_model = ExtraTreesRegressor(random_state=random_state)
    cv_scores = cross_val_score(et_model, x, y, cv=cv, scoring="r2")
    cv_r2 = np.mean(cv_scores)
    
    # Check if CV R2 > threshold AND number of features > 20
    if cv_r2 > threshold and x.shape[1] > 20:
        return 20
    else:
        return 10