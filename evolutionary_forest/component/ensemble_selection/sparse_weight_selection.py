"""
Sparse-weight ensemble selection from the paper:
  - Fit sparse weights over all candidates (nonnegativity, sum-to-one, L1).
  - Select top K models by weight (or nonzeros if count ≈ K).
  - Refit weights on selected subset without sparsity; use for inference.

Supports training-based and validation-based modes.
"""

from collections import defaultdict
from typing import List, Literal, Optional, Tuple, Union, cast

# Threshold for lam_l2 "auto": if 5-fold CV score of 3-NN KNN on (X, y) > this, use lam_l2=0 else 1
LAM_L2_AUTO_KNN_CV_THRESHOLD = 0.8

import numpy as np


def _resolve_lam_l2_auto(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv: int = 5,
    n_neighbors: int = 3,
    threshold: float = LAM_L2_AUTO_KNN_CV_THRESHOLD,
    random_state: Optional[int] = None,
) -> float:
    """
    Resolve lam_l2 for "auto" mode: 5-fold CV of 3-NN KNN on (X, y).
    If mean CV R² > threshold, return 0.0 (no L2); else return 1.0.
    Uses original features and labels, once per evolution process.
    """
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.neighbors import KNeighborsRegressor
    except ImportError:
        return 0.0
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()
    if len(y) < cv * 2:
        return 0.0
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    scores = cross_val_score(knn, X, y, cv=cv, scoring="r2")
    mean_r2 = float(np.mean(scores))
    return 0.0 if mean_r2 > threshold else 0.1


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
    lam_l2: float = 0.0,
) -> np.ndarray:
    """
    Solve: min  L(P @ w, y) + lam * ||w||_1 + lam_l2 * ||w||_2^2  s.t. w >= 0, sum(w)=1, and optionally w_i <= max_weight.
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
        l2 = lam_l2 * np.sum(w ** 2) if lam_l2 != 0 else 0.0
        return fit_loss + l1 + l2

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
    lam_l2: float = 0.0,
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
    lam: float = 0.01,
    K: int = 20,
    equal_weight: bool = False,
    loss: Literal["squared", "absolute", "huber"] = "squared",
    huber_delta: float = 1.0,
    max_weight: Optional[float] = None,
    lam_l2: float = 0.0,
    mode: Literal["sparse", "equal_top", "adaptive"] = "sparse",
    stability_n_bootstrap: int = 0,
    stability_fraction: float = 0.5,
    random_state: Optional[int] = None,
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
        L1 penalty for first-stage sparsity (used when mode is "sparse" or "adaptive").
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
    lam_l2 : float or "auto", default=0.0
        L2 penalty on weights (elastic net in Stage 1, shrinkage in Stage 2 refit).
        0 disables L2. If "auto", resolve once for the whole evolution: 5-fold CV of
        3-NN KNN on original features and labels; use 0 if mean R² > 0.8 else 1.
    mode : {"sparse", "equal_top", "adaptive"}, default="sparse"
        "sparse": L1-regularized sparse weights, then refit or equal on selected.
        "equal_top": select top K models by lowest error (MSE), equal weights.
        "adaptive": same as "sparse" (uses sparse path).
    stability_n_bootstrap : int, default=0
        If > 0, run stability selection: fit sparse weights on this many bootstrap
        samples and keep models with nonzero weight in >= stability_fraction of runs.
        0 disables stability selection.
    stability_fraction : float, default=0.5
        Minimum fraction of bootstrap runs in which a model must be selected (nonzero
        weight) to be kept. Used only when stability_n_bootstrap > 0.
    random_state : int, optional
        Seed for bootstrap sampling when stability_n_bootstrap > 0.

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
    use_sparse = mode in ("sparse", "adaptive")

    if not use_sparse:
        return _top_low_error_equal_weight(P, y_flat, K)

    if stability_n_bootstrap > 0:
        # Stability selection: bootstrap sparse fit, keep models selected in >= stability_fraction of runs
        rng = np.random.default_rng(random_state)
        vote_count = np.zeros(n_models)
        for _ in range(stability_n_bootstrap):
            boot_idx = rng.integers(0, n_samples, size=n_samples)
            P_b = P[boot_idx]
            y_b = y_flat[boot_idx]
            w_b = _sparse_weight_fit(
                P_b, y_b, lam,
                loss=loss, huber_delta=huber_delta, max_weight=max_weight,
                lam_l2=lam_l2,
            )
            vote_count += (w_b > 1e-8).astype(np.float64)
        threshold = stability_fraction * stability_n_bootstrap
        stable_idx = np.where(vote_count >= threshold)[0]
        if len(stable_idx) == 0:
            # Fallback: single-run selection
            w_sparse = _sparse_weight_fit(
                P, y_flat, lam, loss=loss, huber_delta=huber_delta, max_weight=max_weight,
                lam_l2=lam_l2,
            )
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
            P, y_flat, lam, loss=loss, huber_delta=huber_delta, max_weight=max_weight,
            lam_l2=lam_l2,
        )
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
            P_sel, y_flat, loss=loss, huber_delta=huber_delta, max_weight=max_weight,
            lam_l2=lam_l2,
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
    or "adaptive" (same as sparse).
    stability_n_bootstrap > 0 enables stability selection; stability_fraction and random_state
    are passed through. lam_l2 (default 0) adds L2 penalty in Stage 1 and refit.
    lam_l2="auto" resolves L2 once for the whole evolution: 5-fold CV of 3-NN KNN on original
    X and y; use 0 if mean R² > 0.8 else 1 (requires algorithm.X). Supports validation-based
    mode when `algorithm` is provided and has validation_based_ensemble_selection and
    validation data set.
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
        lam_l2: Union[float, Literal["auto"]] = 0.0,
        mode: Literal["sparse", "equal_top", "adaptive"] = "sparse",
        stability_n_bootstrap: int = 0,
        stability_fraction: float = 0.5,
        random_state: Optional[int] = None,
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
        self.lam_l2 = lam_l2
        self.mode = mode
        self.stability_n_bootstrap = stability_n_bootstrap
        self.stability_fraction = stability_fraction
        self.random_state = random_state
        self.algorithm = algorithm
        self.ensemble_weight = defaultdict(float)
        # Resolve lam_l2 once when "auto"; cache for whole evolution
        self._resolved_lam_l2: Optional[float] = None

    def _get_effective_lam_l2(self) -> float:
        """Resolve lam_l2 once when 'auto', then return effective float."""
        if self.lam_l2 != "auto":
            return float(self.lam_l2)
        if self._resolved_lam_l2 is not None:
            return self._resolved_lam_l2
        # Resolve once: 5-fold CV of RF on original X, y; 0 if R² > 0.85 else 1
        if self.algorithm is None:
            raise ValueError(
                "lam_l2='auto' requires algorithm to be provided to SparseWeightHallOfFame."
            )
        X = self.algorithm.X
        if X is None:
            raise ValueError(
                "lam_l2='auto' requires algorithm.X (training features) to be set."
            )
        self._resolved_lam_l2 = _resolve_lam_l2_auto(
            X, self.y, random_state=self.random_state
        )
        return self._resolved_lam_l2

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
        effective_lam_l2 = self._get_effective_lam_l2()
        indices, weights = sparse_weight_ensemble_select(
            P,
            y_fit,
            lam=self.lambda_,
            K=K,
            equal_weight=self.equal_weight,
            loss=self.loss,
            huber_delta=self.huber_delta,
            max_weight=self.max_weight,
            lam_l2=effective_lam_l2,
            mode=cast(Literal["sparse", "equal_top", "adaptive"], self.mode),
            stability_n_bootstrap=self.stability_n_bootstrap,
            stability_fraction=self.stability_fraction,
            random_state=self.random_state,
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
