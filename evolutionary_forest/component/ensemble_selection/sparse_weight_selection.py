"""
Sparse-weight ensemble selection from the paper:
  - Fit sparse-like weights over all candidates (nonnegativity, sum-to-one).
  - Select top K models by weight (or nonzeros if count ≈ K).
  - Renormalize stage-1 weights on selected models to sum to one; use for inference.

Supports training-based and validation-based modes.
"""

from collections import defaultdict
from typing import List, Literal, Optional, Tuple, cast

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
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
    max_weight: Optional[float] = None,
    lam_l2: float = 1e-8,
) -> np.ndarray:
    """
    Solve: min  L(P @ w, y) + lam_l2 * ||w||_2^2  s.t. w >= 0, sum(w)=1, and optionally w_i <= max_weight.
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
        l2 = lam_l2 * np.sum(w ** 2) if lam_l2 != 0 else 0.0
        return fit_loss + l2

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
    lam_l2: float = 1e-8,
) -> np.ndarray:
    """
    Solve: min L(P @ w, y) + lam_l2 * ||w||_2^2  s.t. w >= 0, sum(w)=1, and optionally w_i <= max_weight.
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
            fit_loss = np.mean(res ** 2)
        elif loss == "absolute":
            fit_loss = np.mean(np.abs(res))
        else:
            fit_loss = np.mean(_huber(res, huber_delta))
        l2 = lam_l2 * np.sum(w ** 2) if lam_l2 != 0 else 0.0
        return fit_loss + l2

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
    K: int = 20,
    equal_weight: bool = False,
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
    max_weight: Optional[float] = None,
    lam_l2: float = 1e-8,
    mode: Literal["sparse", "equal_top"] = "sparse",
    stability_n_bootstrap: int = 0,
    stability_fraction: float = 0.5,
    random_state: Optional[int] = None,
    first_stage_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-stage sparse-weight selection, or top low-error equal-weight selection.

    Parameters
    ----------
    P : np.ndarray
        Predictions matrix (n_samples, n_models).
    y : np.ndarray
        Target vector (n_samples,).
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
    lam_l2 : float, default=1e-8
        L2 penalty on weights (elastic net in Stage 1, shrinkage in Stage 2 refit).
    mode : {"sparse", "equal_top"}, default="sparse"
        "sparse": L1-regularized sparse weights, then refit or equal on selected.
        "equal_top": select top K models by lowest error (MSE), equal weights.
    stability_n_bootstrap : int, default=0
        If > 0, run stability selection: fit sparse weights on this many bootstrap
        samples and keep models with nonzero weight in >= stability_fraction of runs.
        0 disables stability selection.
    stability_fraction : float, default=0.5
        Minimum fraction of bootstrap runs in which a model must be selected (nonzero
        weight) to be kept. Used only when stability_n_bootstrap > 0.
    random_state : int, optional
        Seed for bootstrap sampling when stability_n_bootstrap > 0.
    first_stage_only : bool, default=True
        Use Stage 1 sparse weights on selected models and renormalize them
        to sum to 1. This skips Stage 2 refit. When False, an optional
        Stage 2 refit is performed. This takes precedence over equal_weight.

    Returns
    -------
    indices : np.ndarray
        Indices of selected models (length <= n_models).
    weights : np.ndarray
        Weights for selected models (same length as indices). Refit or equal
        depending on equal_weight (sparse path) or always equal (equal_top).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    n_samples, n_models = P.shape
    if n_models == 0:
        return np.array([], dtype=int), np.array([])

    y_flat = np.asarray(y).ravel()
    use_sparse = mode == "sparse"

    if not use_sparse:
        return _top_low_error_equal_weight(P, y_flat, K)

    w_sparse_full: Optional[np.ndarray] = None
    if stability_n_bootstrap > 0:
        # Stability selection: bootstrap sparse fit, keep models selected in >= stability_fraction of runs
        rng = np.random.default_rng(random_state)
        vote_count = np.zeros(n_models)
        for _ in range(stability_n_bootstrap):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            P_b = P[boot_idx]
            y_b = y_flat[boot_idx]
            w_b = _sparse_weight_fit(
                P_b, y_b,
                loss=loss, huber_delta=huber_delta, max_weight=max_weight,
                lam_l2=lam_l2,
            )
            vote_count += (w_b > 1e-8).astype(np.float64)
        threshold = stability_fraction * stability_n_bootstrap
        stable_idx = np.where(vote_count >= threshold)[0]
        if len(stable_idx) == 0:
            # Fallback: single-run selection
            w_sparse = _sparse_weight_fit(
                P, y_flat, loss=loss, huber_delta=huber_delta, max_weight=max_weight,
                lam_l2=lam_l2,
            )
            w_sparse_full = w_sparse
            top_k_idx = np.argsort(w_sparse)[::-1][: min(K, n_models)]
            nonzero_idx = np.where(w_sparse > 1e-8)[0]
            selected_idx = nonzero_idx if len(nonzero_idx) <= K else top_k_idx
            if len(selected_idx) == 0:
                selected_idx = np.array([np.argmax(w_sparse)])
        elif len(stable_idx) <= K:
            selected_idx = stable_idx
        else:
            selected_idx = stable_idx[np.argsort(vote_count[stable_idx])[::-1][:K]]
    else:
        # Stage 1: single-run sparse weights
        w_sparse = _sparse_weight_fit(
            P, y_flat, loss=loss, huber_delta=huber_delta, max_weight=max_weight,
            lam_l2=lam_l2,
        )
        w_sparse_full = w_sparse
        top_k_idx = np.argsort(w_sparse)[::-1][: min(K, n_models)]
        nonzero_idx = np.where(w_sparse > 1e-8)[0]
        if len(nonzero_idx) <= K:
            selected_idx = nonzero_idx
        else:
            selected_idx = top_k_idx
        if len(selected_idx) == 0:
            selected_idx = np.array([np.argmax(w_sparse)])

    # Stage 1 selected weights (with optional Stage 2 refit):
    n_sel = len(selected_idx)
    if first_stage_only:
        if w_sparse_full is None:
            w_sparse_full = _sparse_weight_fit(
                P, y_flat, loss=loss, huber_delta=huber_delta, max_weight=max_weight,
                lam_l2=lam_l2,
            )
        w_selected = np.asarray(w_sparse_full[selected_idx], dtype=float)
        w_selected = np.maximum(w_selected, 0.0)
        w_refit = w_selected / (np.sum(w_selected) + 1e-12)
    elif equal_weight:
        w_refit = np.ones(n_sel) / n_sel
    else:
        P_sel = P[:, selected_idx]
        w_refit = _refit_weights(
            P_sel, y_flat, loss=loss, huber_delta=huber_delta, max_weight=max_weight,
            lam_l2=lam_l2,
        )

    return selected_idx, w_refit


class SparseWeightHallOfFame(HallOfFame):
    """
    Hall of fame that selects and weights individuals using the sparse-weight
    pipeline: L1-regularized weight fitting, top-K selection, then optional
    Stage-2 refit or equal weights. Set equal_weight=True to use equal weights
    (1/n_selected) instead of weighted inference; Stage-2 refit is disabled by
    default.
    Use loss="squared" (MSE), loss="absolute" (MAE), or loss="huber" (Huber).
    huber_delta is used only when loss="huber" (default 1.0).
    mode: "sparse" (default), "equal_top" (top K by lowest error, equal weight).
    stability_n_bootstrap > 0 enables stability selection; stability_fraction and random_state
    are passed through. lam_l2 (default 1e-8) adds L2 penalty in Stage 1 and optional
    Stage 2 refit. first_stage_only=True uses normalized Stage 1 sparse weights on
    selected models.
    """

    def __init__(
        self,
        maxsize: int,
        y: np.ndarray,
        equal_weight: bool = False,
        loss: Literal["squared", "absolute", "huber"] = "squared",
        huber_delta: float = 1.0,
        max_weight: Optional[float] = None,
        lam_l2: float = 1e-8,
        mode: Literal["sparse", "equal_top"] = "sparse",
        stability_n_bootstrap: int = 0,
        stability_fraction: float = 0.5,
        random_state: Optional[int] = None,
        first_stage_only: bool = True,
        algorithm=None,
        similar=eq,
        **kwargs,
    ):
        super().__init__(maxsize, similar)
        self.y = np.asarray(y).ravel()
        self.equal_weight = equal_weight
        self.loss: Literal["squared", "absolute", "huber"] = loss
        self.huber_delta = huber_delta
        self.max_weight = max_weight
        self.lam_l2 = lam_l2
        self.mode = mode
        self.stability_n_bootstrap = stability_n_bootstrap
        self.stability_fraction = stability_fraction
        self.random_state = random_state
        self.first_stage_only = first_stage_only
        self.algorithm = algorithm
        self.ensemble_weight = defaultdict(float)
        self.algorithm = algorithm

    def update(self, population: List) -> None:
        if not population:
            return

        # Pool = previous ensemble (current HOF) + current population
        previous_ensemble = list(self)
        candidates = previous_ensemble + list(population)

        # Build prediction matrix P (n_samples, n_candidates)
        use_validation = (
            self.algorithm is not None
            and self.algorithm.validation_based_ensemble_selection > 0
        )

        if use_validation:
            # Validation-based: get predictions on validation set
            # individual_prediction returns (n_individuals, n_samples)
            algo = self.algorithm
            assert algo is not None
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
            K=K,
            equal_weight=self.equal_weight,
            loss=self.loss,
            huber_delta=self.huber_delta,
            max_weight=self.max_weight,
            lam_l2=self.lam_l2,
            mode=cast(Literal["sparse", "equal_top"], self.mode),
            stability_n_bootstrap=self.stability_n_bootstrap,
            stability_fraction=self.stability_fraction,
            random_state=self.random_state,
            first_stage_only=self.first_stage_only,
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
