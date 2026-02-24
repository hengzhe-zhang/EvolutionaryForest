"""
Sparse-weight ensemble selection from the paper:
  - Fit sparse weights over all candidates (nonnegativity, sum-to-one, L1).
  - Select top K models by weight (or nonzeros if count ≈ K).
  - Refit weights on selected subset without sparsity; use for inference.

Supports training-based and validation-based modes.
"""

from collections import defaultdict
from typing import List, Literal, Tuple

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
) -> np.ndarray:
    """
    Solve: min_{w >= 0, sum(w)=1}  L(P @ w, y) + lam * ||w||_1.
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

    # Bounds: w >= 0
    bounds = [(0.0, None)] * n_models
    # Constraint: sum(w) = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    w0 = np.ones(n_models) / n_models
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        # Fallback: uniform weights
        return np.ones(n_models) / n_models
    return np.maximum(res.x, 0.0) / (np.sum(np.maximum(res.x, 0.0)) + 1e-12)


def _refit_weights(
    P: np.ndarray,
    y: np.ndarray,
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
) -> np.ndarray:
    """
    Solve: min_{w >= 0, sum(w)=1}  L(P @ w, y).
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

    bounds = [(0.0, None)] * n_sel
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w0 = np.ones(n_sel) / n_sel
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        return np.ones(n_sel) / n_sel
    w = np.maximum(res.x, 0.0)
    return w / (np.sum(w) + 1e-12)


def sparse_weight_ensemble_select(
    P: np.ndarray,
    y: np.ndarray,
    lam: float = 0.01,
    K: int = 20,
    equal_weight: bool = False,
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage sparse-weight selection.

    Parameters
    ----------
    P : np.ndarray
        Predictions matrix (n_samples, n_models).
    y : np.ndarray
        Target vector (n_samples,).
    lam : float
        L1 penalty for first-stage sparsity.
    K : int
        Maximum number of models to select. Ensemble size is always <= K (can be less).
    equal_weight : bool, default=False
        If True, use equal weights (1/n_selected) for selected models instead of
        refitting weights on the selected subset. Can improve generalization.
    loss : {"squared", "absolute", "huber"}, default="squared"
        Loss for weight fitting: "squared" = MSE, "absolute" = MAE, "huber" = Huber.
    huber_delta : float, default=1.0
        Delta for Huber loss (used only when loss="huber").

    Returns
    -------
    indices : np.ndarray
        Indices of selected models (length <= n_models).
    weights : np.ndarray
        Weights for selected models (same length as indices). Refit or equal
        depending on equal_weight.
    """
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    n_samples, n_models = P.shape
    if n_models == 0:
        return np.array([], dtype=int), np.array([])

    # Stage 1: sparse weights
    w_sparse = _sparse_weight_fit(P, y, lam, loss=loss, huber_delta=huber_delta)

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
        w_refit = _refit_weights(P_sel, y, loss=loss, huber_delta=huber_delta)

    return selected_idx, w_refit


class SparseWeightHallOfFame(HallOfFame):
    """
    Hall of fame that selects and weights individuals using the sparse-weight
    pipeline: L1-regularized weight fitting, top-K selection, then refit or
    equal weights. Set equal_weight=True to use equal weights (1/n_selected)
    instead of refitting; can improve generalization.
    Use loss="squared" (MSE), loss="absolute" (MAE), or loss="huber" (Huber).
    huber_delta is used only when loss="huber" (default 1.0).
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
