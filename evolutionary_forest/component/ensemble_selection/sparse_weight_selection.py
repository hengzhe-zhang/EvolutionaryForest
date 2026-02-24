"""
Sparse-weight ensemble selection from the paper:
  - Fit sparse weights over all candidates (nonnegativity, sum-to-one, L1).
  - Select top K models by weight (or nonzeros if count ≈ K).
  - Refit weights on selected subset without sparsity; use for inference.

Supports training-based and validation-based modes.
"""

from collections import defaultdict
from typing import List, Literal, Optional, Tuple, cast

# Threshold for adaptive mode: sparse weights when n_unique_y > ADAPTIVE_N_UNIQUE_THRESHOLD
ADAPTIVE_N_UNIQUE_THRESHOLD = 50

import numpy as np


def _huber(res: np.ndarray, delta: float) -> np.ndarray:
    """Huber loss per element: 0.5*r^2 if |r|<=delta else delta*(|r|-0.5*delta)."""
    abs_r = np.abs(res)
    return np.where(abs_r <= delta, 0.5 * res ** 2, delta * (abs_r - 0.5 * delta))
from deap.tools import HallOfFame
from operator import eq
from scipy.optimize import minimize

from evolutionary_forest.component.primitive_functions import individual_to_tuple


def _sparse_weight_fit(
    P: np.ndarray,
    y: np.ndarray,
    lam: float,
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
    max_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Solve: min  L(P @ w, y) + lam * ||w||_1  s.t. w >= 0, sum(w)=1, and optionally w_i <= max_weight.
    L is MSE (squared), MAE (absolute), or mean Huber (huber).
    P: (n_samples, n_models), y: (n_samples,).
    Returns w: (n_models,).
    """
    n_samples, n_models = P.shape
    y = np.asarray(y).ravel()

    def obj(w: np.ndarray) -> float:
        pred = P @ w
        res = pred - y
        if loss == "squared":
            fit_loss = np.mean(res ** 2)
        elif loss == "absolute":
            fit_loss = np.mean(np.abs(res))
        else:
            fit_loss = np.mean(_huber(res, huber_delta))
        l1 = lam * np.sum(np.abs(w))
        return fit_loss + l1

    # Bounds: 0 <= w_i <= max_weight (if set)
    ub = max_weight if max_weight is not None else None
    bounds = [(0.0, ub)] * n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    w0 = np.ones(n_models) / n_models
    if max_weight is not None:
        w0 = np.minimum(w0, max_weight)
        w0 /= w0.sum()
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        w = np.ones(n_models) / n_models
    else:
        w = np.maximum(res.x, 0.0)
    if max_weight is not None:
        w = np.minimum(w, max_weight)
    return w / (np.sum(w) + 1e-12)


def _refit_weights(
    P: np.ndarray,
    y: np.ndarray,
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
    max_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Solve: min L(P @ w, y)  s.t. w >= 0, sum(w)=1, and optionally w_i <= max_weight.
    L is MSE (squared), MAE (absolute), or mean Huber (huber).
    P: (n_samples, n_selected), y: (n_samples,).
    Returns w: (n_selected,).
    """
    n_samples, n_sel = P.shape
    y = np.asarray(y).ravel()
    if n_sel == 0:
        return np.array([])

    def obj(w: np.ndarray) -> float:
        res = P @ w - y
        if loss == "squared":
            return np.mean(res ** 2)
        if loss == "absolute":
            return np.mean(np.abs(res))
        return np.mean(_huber(res, huber_delta))

    ub = max_weight if max_weight is not None else None
    bounds = [(0.0, ub)] * n_sel
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n_sel) / n_sel
    if max_weight is not None:
        w0 = np.minimum(w0, max_weight)
        w0 /= w0.sum()
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        w = np.ones(n_sel) / n_sel
    else:
        w = np.maximum(res.x, 0.0)
    if max_weight is not None:
        w = np.minimum(w, max_weight)
    return w / (np.sum(w) + 1e-12)


def _top_low_error_equal_weight(
    P: np.ndarray, y: np.ndarray, K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top K models by lowest MSE on (P, y), with equal weights 1/K.
    """
    n_models = P.shape[1]
    K = min(K, n_models)
    if K <= 0:
        return np.array([], dtype=int), np.array([])
    y = np.asarray(y).ravel()
    mse_per_model = np.mean((P - y[:, None]) ** 2, axis=0)
    selected_idx = np.argsort(mse_per_model)[:K]
    weights = np.ones(K) / K
    return selected_idx, weights


def sparse_weight_ensemble_select(
    P: np.ndarray,
    y: np.ndarray,
    lam: float = 0.01,
    K: int = 20,
    equal_weight: bool = False,
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
    max_weight: Optional[float] = None,
    mode: Literal["sparse", "equal_top", "adaptive"] = "sparse",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage sparse-weight selection, or top low-error equal-weight selection.

    Parameters
    ----------
    P : np.ndarray
        Predictions matrix (n_samples, n_models).
    y : np.ndarray
        Target vector (n_samples,).
    lam : float
        L1 penalty for first-stage sparsity (used when mode is "sparse" or "adaptive" and n_unique_y > threshold).
    K : int
        Maximum number of models to select. Ensemble size is always <= K (can be less).
    equal_weight : bool, default=False
        If True, use equal weights (1/n_selected) for selected models instead of
        refitting weights on the selected subset. Can improve generalization.
    loss : {"squared", "absolute", "huber"}, default="squared"
        Loss for weight fitting: "squared" = MSE, "absolute" = MAE, "huber" = Huber.
    huber_delta : float, default=1.0
        Delta for Huber loss (used only when loss="huber").
    max_weight : float, optional
        Upper bound on each weight (e.g. 0.1). If set, enforces w_i <= max_weight.
        Requires at least 1/max_weight models to sum to 1.
    mode : {"sparse", "equal_top", "adaptive"}, default="sparse"
        "sparse": L1-regularized sparse weights, then refit or equal on selected.
        "equal_top": select top K models by lowest error (MSE), equal weights.
        "adaptive": if number of unique y values > ADAPTIVE_N_UNIQUE_THRESHOLD use sparse path;
        otherwise use equal_top (top low-error models, equal weight).

    Returns
    -------
    indices : np.ndarray
        Indices of selected models (length <= n_models).
    weights : np.ndarray
        Weights for selected models (same length as indices). Refit or equal
        depending on equal_weight (sparse path) or always equal (equal_top/adaptive fallback).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    n_samples, n_models = P.shape
    if n_models == 0:
        return np.array([], dtype=int), np.array([])

    y_flat = np.asarray(y).ravel()
    use_sparse = mode == "sparse" or (
        mode == "adaptive"
        and len(np.unique(y_flat)) > ADAPTIVE_N_UNIQUE_THRESHOLD
    )

    if not use_sparse:
        return _top_low_error_equal_weight(P, y_flat, K)

    # Stage 1: sparse weights
    w_sparse = _sparse_weight_fit(
        P, y_flat, lam, loss=loss, huber_delta=huber_delta, max_weight=max_weight
    )

    # Selection: at most K models; prefer L1 nonzeros when count <= K, else top K by weight
    top_k_idx = np.argsort(w_sparse)[::-1][: min(K, n_models)]
    nonzero_idx = np.where(w_sparse > 1e-8)[0]
    if len(nonzero_idx) <= K:
        selected_idx = nonzero_idx
    else:
        selected_idx = top_k_idx

    if len(selected_idx) == 0:
        selected_idx = np.array([np.argmax(w_sparse)])

    # Stage 2: weights on selected subset
    n_sel = len(selected_idx)
    if equal_weight:
        w_refit = np.ones(n_sel) / n_sel
    else:
        P_sel = P[:, selected_idx]
        w_refit = _refit_weights(
            P_sel, y_flat, loss=loss, huber_delta=huber_delta, max_weight=max_weight
        )

    return selected_idx, w_refit


class SparseWeightHallOfFame(HallOfFame):
    """
    Hall of fame that selects and weights individuals using the sparse-weight
    pipeline: L1-regularized weight fitting, top-K selection, then refit or
    equal weights. Set equal_weight=True to use equal weights (1/n_selected)
    instead of refitting; can improve generalization.
    Use loss="squared" (MSE), loss="absolute" (MAE), or loss="huber" (Huber).
    huber_delta is used only when loss="huber" (default 1.0).
    mode: "sparse" (default), "equal_top" (top K by lowest error, equal weight),
    or "adaptive" (sparse when number of unique y > ADAPTIVE_N_UNIQUE_THRESHOLD, else equal_top).
    Supports validation-based mode when `algorithm` is provided and has
    validation_based_ensemble_selection and validation data set.
    """

    def __init__(
        self,
        maxsize: int,
        y: np.ndarray,
        lambda_: float = 0.01,
        equal_weight: bool = False,
        loss: Literal["squared", "absolute", "huber"] = "squared",
        huber_delta: float = 1.0,
        max_weight: Optional[float] = None,
        mode: Literal["sparse", "equal_top", "adaptive"] = "sparse",
        algorithm=None,
        similar=eq,
        **kwargs,
    ):
        super().__init__(maxsize, similar)
        self.y = np.asarray(y).ravel()
        self.lambda_ = lambda_
        self.equal_weight = equal_weight
        self.loss: Literal["squared", "absolute", "huber"] = loss
        self.huber_delta = huber_delta
        self.max_weight = max_weight
        self.mode = mode
        self.algorithm = algorithm
        self.ensemble_weight = defaultdict(float)

    def update(self, population: List) -> None:
        if not population:
            return

        # Pool = previous ensemble (current HOF) + current population
        previous_ensemble = list(self)
        candidates = previous_ensemble + list(population)

        # Build prediction matrix P (n_samples, n_candidates)
        use_validation = (
            self.algorithm is not None
            and getattr(self.algorithm, "validation_based_ensemble_selection", 0) > 0
            and hasattr(self.algorithm, "des_valid_x")
            and hasattr(self.algorithm, "des_valid_y")
        )

        if use_validation:
            # Validation-based: get predictions on validation set
            # individual_prediction returns (n_individuals, n_samples)
            algo = self.algorithm
            assert algo is not None and hasattr(algo, "des_valid_x")
            valid_preds = algo.individual_prediction(algo.des_valid_x, candidates)
            P = np.asarray(valid_preds).T  # (n_samples, n_candidates)
            y_fit = np.asarray(algo.des_valid_y).ravel()
        else:
            # Training-based: use predicted_values from pool
            P = np.array([ind.predicted_values.flatten() for ind in candidates]).T
            y_fit = self.y

        n_samples, n_models = P.shape
        if n_models == 0:
            return

        # Run sparse-weight selection on pool; K = ensemble_size (maxsize)
        K = min(self.maxsize, n_models)
        indices, weights = sparse_weight_ensemble_select(
            P,
            y_fit,
            lam=self.lambda_,
            K=K,
            equal_weight=self.equal_weight,
            loss=self.loss,
            huber_delta=self.huber_delta,
            max_weight=self.max_weight,
            mode=cast(Literal["sparse", "equal_top", "adaptive"], self.mode),
        )

        # Map indices to candidates (pool = previous ensemble + population)
        selected = [candidates[i] for i in indices]
        self.ensemble_weight.clear()
        for ind, w in zip(selected, weights):
            key = individual_to_tuple(ind)
            self.ensemble_weight[key] = float(w)

        # Replace hall of fame with selected individuals
        self.clear()
        for ind in selected:
            self.insert(ind)
