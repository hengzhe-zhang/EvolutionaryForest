from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import sympy as sp
from sympy import lambdify
from sympy.parsing.sympy_parser import parse_expr


def sympy_expr_to_string(expr: sp.Expr) -> str:
    return sp.srepr(expr)


def sympy_expr_from_string(expr_text: str) -> sp.Expr:
    return cast(sp.Expr, sp.sympify(expr_text))


def _safe_eval_expr(expr_text: str, n_features: int) -> sp.Expr:
    expr_text = expr_text.strip().replace("^", "**")
    expr_text = expr_text.replace("zoo", "oo")
    round_fn = sp.Function("Round")
    max_fn = sp.Function("Max")
    min_fn = sp.Function("Min")
    pdiv_fn = sp.Function("protected_division")
    symbols: dict[str, Any] = {f"x{i}": sp.Symbol(f"x{i}") for i in range(n_features)}
    symbols.update(
        {
            "Max": max_fn,
            "Min": min_fn,
            "protected_division": pdiv_fn,
            "Abs": sp.Abs,
            "round": round_fn,
            "Round": round_fn,
            "sqrt": sp.sqrt,
        }
    )
    return parse_expr(expr_text, local_dict=symbols, evaluate=False)


def _pretty_expr(expr: sp.Expr, var_names: list[str]) -> str:
    if not var_names:
        return str(expr)
    return str(
        expr.xreplace(
            {
                sp.Symbol(f"x{i}"): sp.Symbol(name)
                for i, name in enumerate(var_names)
                if f"x{i}" in str(expr)
            }
        )
    )


def symbolic_from_estimator(
    estimator: Any,
    n_features: int,
    var_names: list[str],
) -> tuple[str, sp.Expr]:
    if not hasattr(estimator, "model"):
        raise RuntimeError("Estimator does not expose symbolic form via model().")
    expr_text = str(estimator.model())
    expr = _safe_eval_expr(expr_text, n_features)
    return _pretty_expr(expr, var_names), expr


def _coerce_output_array(values: Any, n_samples: int) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError):
        cur = values
        for _ in range(12):
            if isinstance(cur, np.ndarray):
                if cur.size == 0:
                    raise ValueError("Empty symbolic output.")
                try:
                    arr = np.asarray(cur, dtype=float)
                    break
                except (ValueError, TypeError):
                    cur = cur.ravel()[0]
                    continue
            if isinstance(cur, (list, tuple, sp.Matrix)):
                if len(cur) == 0:
                    raise ValueError("Empty symbolic output.")
                try:
                    arr = np.asarray(cur, dtype=float)
                    break
                except (ValueError, TypeError):
                    cur = cur[0]
                    continue
            arr = np.full(n_samples, float(cur), dtype=float)
            break
        else:
            raise ValueError("Could not coerce symbolic output to numeric array.")

    if arr.ndim == 0:
        return np.full(n_samples, float(arr), dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] == 1:
            return np.full(n_samples, float(arr[0]), dtype=float)
        return arr
    if arr.ndim == 2 and arr.shape[0] == n_samples:
        if arr.shape[1] == 1:
            return arr[:, 0]
        raise ValueError(f"Unexpected output matrix shape {arr.shape}.")
    if arr.ndim == 2 and arr.shape[1] == n_samples:
        if arr.shape[0] == 1:
            return arr[0]
        raise ValueError(f"Unexpected output matrix shape {arr.shape}.")
    return np.asarray(values, dtype=float).reshape(-1)


def _np_nary_max(*args: Any) -> np.ndarray:
    if len(args) == 0:
        raise ValueError("Max requires at least one argument.")
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
    out = arrays[0]
    for arr in arrays[1:]:
        out = np.maximum(out, arr)
    return out


def _np_nary_min(*args: Any) -> np.ndarray:
    if len(args) == 0:
        raise ValueError("Min requires at least one argument.")
    arrays = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
    out = arrays[0]
    for arr in arrays[1:]:
        out = np.minimum(out, arr)
    return out


def _np_protected_div(a: Any, b: Any, eps: float = 1e-10) -> np.ndarray:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)
    safe_sign = np.where(b_arr >= 0, 1.0, -1.0)
    safe_den = safe_sign * np.sqrt(b_arr * b_arr + eps)
    return a_arr / safe_den


def predict_sympy_expression(expr: sp.Expr, X: np.ndarray) -> np.ndarray:
    start = time.time()
    symbols = [sp.Symbol(f"x{i}") for i in range(X.shape[1])]
    fn = lambdify(
        symbols,
        expr,
        modules=[
            {
                "Max": _np_nary_max,
                "Min": _np_nary_min,
                "protected_division": _np_protected_div,
                "SafeDiv": _np_protected_div,
                "round": np.round,
                "Round": np.round,
            },
            "numpy",
        ],
    )

    cols = [X[:, i] for i in range(X.shape[1])]
    try:
        with np.errstate(all="ignore"):
            out = fn(*cols)
    except ValueError:
        ufunc = np.frompyfunc(fn, X.shape[1], 1)
        with np.errstate(all="ignore"):
            out = ufunc(*cols)

    out = _coerce_output_array(out, X.shape[0])
    out = np.asarray(out, dtype=float)
    n_invalid = int(np.size(out) - np.count_nonzero(np.isfinite(out)))
    if n_invalid:
        out = np.where(np.isfinite(out), out, np.nan)
    print(
        f"[INFO] Symbolic eval done: samples={len(out)}, elapsed={time.time() - start:.2f}s"
    )
    return out
