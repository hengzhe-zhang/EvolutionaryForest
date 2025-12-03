import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._base import LinearModel
from sklearn.tree import BaseDecisionTree


def _create_shap_explainer(base_learner, background_data):
    """Create appropriate SHAP explainer for the base learner."""
    import shap

    if isinstance(base_learner, (LinearModel, LogisticRegression)):
        return shap.LinearExplainer(base_learner, background_data)
    elif isinstance(base_learner, BaseDecisionTree):
        return shap.TreeExplainer(base_learner, background_data)
    else:
        raise Exception(f"Unsupported model type for SHAP: {type(base_learner)}")


def _calculate_shap_values(explainer, base_learner, data):
    """Calculate SHAP values for the given data."""
    if isinstance(base_learner, BaseDecisionTree):
        return explainer.shap_values(data)[0]
    else:
        return explainer.shap_values(data)


def _get_cv_data_splits(cv, X, Y, estimator_id):
    """Get train and test data splits for a specific CV fold."""
    split_fold = list(cv.split(X, Y))
    train_id, test_id = split_fold[estimator_id][0], split_fold[estimator_id][1]
    return X[train_id], X[test_id], Y[test_id]


def calculate_shap_importance(estimators, X, Y, cv=None):
    """
    Calculate SHAP values for feature importance from estimators.
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
        Modifies estimators in-place by adding shap_values attribute to base learners
    """
    is_cv = cv is not None and len(estimators) == cv.n_splits

    for id, estimator in enumerate(estimators):
        base_learner = estimator["Ridge"]

        if is_cv:
            background_data, test_data, _ = _get_cv_data_splits(cv, X, Y, id)
        else:
            background_data, test_data = X, X

        explainer = _create_shap_explainer(base_learner, background_data)
        feature_importance = _calculate_shap_values(explainer, base_learner, test_data)
        base_learner.shap_values = np.abs(feature_importance).mean(axis=0)


def calculate_permutation_importance_from_estimators(estimators, X, Y, cv=None):
    """
    Calculate permutation importance from estimators.
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
        Modifies estimators in-place by adding permutation_importance_values attribute to base learners
    """
    is_cv = cv is not None and len(estimators) == cv.n_splits

    for id, estimator in enumerate(estimators):
        base_learner = estimator["Ridge"]

        if is_cv:
            _, test_data, test_y = _get_cv_data_splits(cv, X, Y, id)
        else:
            test_data, test_y = X, Y

        r = permutation_importance(
            base_learner,
            test_data,
            test_y,
            n_jobs=1,
            n_repeats=1,
        )
        base_learner.permutation_importance_values = np.abs(r.importances_mean)
