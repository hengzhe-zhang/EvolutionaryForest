import numpy as np
from sklearn.base import clone
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from typing import Any, cast


def _safe_r2_score(y_true, y_pred) -> float:
    """Compute R² on finite rows only; fall back safely when invalid."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.ndim <= 1 and y_pred_arr.ndim <= 1:
        finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    else:
        y_true_2d = y_true_arr if y_true_arr.ndim > 1 else y_true_arr.reshape(-1, 1)
        y_pred_2d = y_pred_arr if y_pred_arr.ndim > 1 else y_pred_arr.reshape(-1, 1)
        finite_mask = np.all(np.isfinite(y_true_2d), axis=1) & np.all(
            np.isfinite(y_pred_2d), axis=1
        )

    # R² is undefined with less than two valid samples.
    if np.count_nonzero(finite_mask) < 2:
        return 0.0

    return float(r2_score(y_true_arr[finite_mask], y_pred_arr[finite_mask]))


def calculate_r2_drop_importance(base_learner, X, y, X_eval=None, y_eval=None):
    """
    Calculate refit-based R² drop importance for linear models.

    For each feature j:
    1. Fit a full model with all features.
    2. Fit a reduced model without feature j.
    3. Compute ΔR²_j = R²_full - R²_reduced on evaluation data.

    Parameters
    ----------
    base_learner : sklearn linear model
        Fitted linear model (Ridge, LinearRegression, etc.); only used as a
        template for cloning hyperparameters.
    X : np.ndarray
        Training feature matrix used for refitting
    y : np.ndarray
        Training target values
    X_eval : np.ndarray, optional
        Evaluation feature matrix. If None, X is used.
    y_eval : np.ndarray, optional
        Evaluation target values. If None, y is used.

    Returns
    -------
    np.ndarray
        R² drop importance values for each feature
    """
    if not isinstance(base_learner, (LinearModel, LogisticRegression)):
        raise ValueError("R² drop importance only supports linear models")

    X = np.asarray(X)
    y = np.asarray(y)
    if X_eval is None or y_eval is None:
        X_eval = X
        y_eval = y
    else:
        X_eval = np.asarray(X_eval)
        y_eval = np.asarray(y_eval)

    _, p = X.shape
    if p <= 1:
        raise ValueError(
            "R² drop importance requires at least two features for leave-one-out refitting"
        )

    learner_template = cast(Any, base_learner)
    full_model = cast(Any, clone(learner_template))
    full_model.fit(X, y)
    baseline_r2 = _safe_r2_score(y_eval, full_model.predict(X_eval))
    r2_drop = np.zeros(p, dtype=float)

    for feature_idx in range(p):
        X_reduced_train = np.delete(X, feature_idx, axis=1)
        X_reduced_eval = np.delete(X_eval, feature_idx, axis=1)
        reduced_model = cast(Any, clone(learner_template))
        reduced_model.fit(X_reduced_train, y)
        reduced_r2 = _safe_r2_score(y_eval, reduced_model.predict(X_reduced_eval))
        r2_drop[feature_idx] = baseline_r2 - reduced_r2

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
    is_cv = cv is not None and len(estimators) == cv.n_splits

    if is_cv:
        assert cv is not None
        split_fold = list(cv.split(X, Y))
    else:
        split_fold = None

    for id, estimator in enumerate(estimators):
        base_learner = estimator["Ridge"]

        if is_cv:
            assert split_fold is not None
            train_id, test_id = split_fold[id][0], split_fold[id][1]
            train_data, train_y = X[train_id], Y[train_id]
            test_data, test_y = X[test_id], Y[test_id]
        else:
            train_data, train_y = X, Y
            test_data, test_y = X, Y

        r2_drop_importance = calculate_r2_drop_importance(
            base_learner,
            train_data,
            train_y,
            X_eval=test_data,
            y_eval=test_y,
        )
        base_learner.r2_drop_importance_values = r2_drop_importance
