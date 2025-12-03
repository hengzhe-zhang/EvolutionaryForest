import numpy as np
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


def calculate_r2_drop_importance(base_learner, X, y):
    """
    Calculate closed-form R² drop importance for linear models.

    Formula: ΔR²_j = (1 - R²) · (t_j² / (t_j² + (n - p - 1)))

    Where:
    - R² is the overall R² of the model
    - t_j is the t-statistic for variable j
    - n is the number of samples
    - p is the number of features/parameters

    Parameters
    ----------
    base_learner : sklearn linear model
        Fitted linear model (Ridge, LinearRegression, etc.)
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values

    Returns
    -------
    np.ndarray
        R² drop importance values for each feature
    """
    if not isinstance(base_learner, (LinearModel, LogisticRegression)):
        raise ValueError("R² drop importance only supports linear models")

    n, p = X.shape
    y_pred = base_learner.predict(X)
    r2 = r2_score(y, y_pred)

    if isinstance(base_learner, LogisticRegression):
        coef = (
            base_learner.coef_[0]
            if len(base_learner.coef_.shape) > 1
            else base_learner.coef_
        )
    else:
        coef = base_learner.coef_

    if len(coef.shape) > 1:
        coef = np.max(np.abs(coef), axis=0)

    residuals = y - y_pred
    mse = np.mean(residuals**2)

    if mse == 0:
        return np.ones(p) / p

    X_with_intercept = np.column_stack([np.ones(n), X])
    try:
        XtX_inv = np.linalg.pinv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(mse * np.diag(XtX_inv)[1:])
    except Exception:
        se = np.ones(p) * np.sqrt(mse)

    se = np.maximum(se, 1e-10)
    t_stats = np.abs(coef) / se

    df = n - p - 1
    if df <= 0:
        df = 1

    r2_drop = (1 - r2) * (t_stats**2 / (t_stats**2 + df))

    return np.abs(r2_drop)


def calculate_r2_drop_importance_from_estimators(estimators, X, Y, cv=None):
    """
    Calculate R² drop importance from estimators.
    Supports both CV and non-CV scenarios (e.g., RidgeCV).

    Parameters
    ----------
    estimators : list
        List of fitted estimator pipelines (can be single estimator for non-CV)
    X : np.ndarray
        Transformed feature matrix (GP tree outputs)
    Y : np.ndarray
        Target values
    cv : sklearn.model_selection.BaseCrossValidator, optional
        Cross-validation splitter (None for non-CV scenarios)

    Returns
    -------
    None
        Modifies estimators in-place by adding r2_drop_importance_values attribute to base learners
    """
    from evolutionary_forest.utility.estimator_feature_importance import (
        _get_cv_data_splits,
    )

    is_cv = cv is not None and len(estimators) == cv.n_splits

    for id, estimator in enumerate(estimators):
        base_learner = estimator["Ridge"]

        if is_cv:
            _, test_data, test_y = _get_cv_data_splits(cv, X, Y, id)
        else:
            test_data, test_y = X, Y

        r2_drop_importance = calculate_r2_drop_importance(
            base_learner, test_data, test_y
        )
        base_learner.r2_drop_importance_values = r2_drop_importance
