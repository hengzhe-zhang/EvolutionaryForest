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
