import torch
import numpy as np
import time
import copy as _copy
from deap import base, creator, gp, tools, algorithms
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
from deap.gp import PrimitiveTree, Primitive, Terminal
import functools
import operator
from sklearn.linear_model import Ridge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Safe tensor-aware primitives ----------


def _ensure_tensor(x, ref: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cast numbers/ndarrays to a torch tensor on the right device and dtype.
    If ref is provided, match its shape by expanding/broadcasting when needed.
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x).float()
    else:
        try:
            t = torch.tensor(float(x), dtype=torch.float32)
        except Exception:
            t = torch.tensor(0.0, dtype=torch.float32)
    t = t.to(ref.device if isinstance(ref, torch.Tensor) else device)
    # Expand scalar to match reference length if ref is 1D tensor
    if isinstance(ref, torch.Tensor) and ref.ndim >= 1 and t.ndim == 0:
        t = t.expand_as(ref)
    return t


def _to_numpy_safe(obj):
    """Recursively convert torch tensors inside containers to numpy arrays.
    Avoids non-leaf tensor deepcopy issues in DEAP's HallOfFame.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().float().numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_numpy_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_numpy_safe(v) for v in obj)
    return obj


# Monkey-patch PrimitiveTree deepcopy to sanitize tensors in __dict__
_def_deepcopy = PrimitiveTree.__deepcopy__


def _safe_PrimitiveTree_deepcopy(self, memo):
    new = type(self)(self)
    sanitized = {k: _to_numpy_safe(v) for k, v in self.__dict__.items()}
    new.__dict__.update(_copy.deepcopy(sanitized, memo))
    return new


PrimitiveTree.__deepcopy__ = _safe_PrimitiveTree_deepcopy


def safe_add(a, b):
    a = _ensure_tensor(a)
    b = _ensure_tensor(b, a)
    return a + b


def safe_sub(a, b):
    a = _ensure_tensor(a)
    b = _ensure_tensor(b, a)
    return a - b


def safe_mul(a, b):
    a = _ensure_tensor(a)
    b = _ensure_tensor(b, a)
    return a * b


def safe_div(a, b):
    a = _ensure_tensor(a)
    b = _ensure_tensor(b, a)
    eps = torch.tensor(1e-6, device=a.device, dtype=a.dtype)
    denom = torch.where(torch.abs(b) > eps, b, torch.ones_like(b))
    return a / denom


def safe_neg(x):
    x = _ensure_tensor(x)
    return -x


def safe_abs(x):
    x = _ensure_tensor(x)
    return torch.abs(x)


def safe_sqrt(x):
    x = _ensure_tensor(x)
    return torch.sqrt(torch.abs(x))


def safe_log(x):
    x = _ensure_tensor(x)
    return torch.log(torch.abs(x) + 1e-6)


class SklearnRegressionWrapper:
    """Wraps torch operations for sklearn-like regression interface"""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.pset = None
        self.toolbox = None
        self.scaler = StandardScaler()

    def _build_primitives(self) -> gp.PrimitiveSet:
        pset = gp.PrimitiveSet("MAIN", arity=self.n_features)

        # Use safe primitives that handle scalars/tensors cleanly
        pset.addPrimitive(safe_add, 2, name="add")
        pset.addPrimitive(safe_sub, 2, name="sub")
        pset.addPrimitive(safe_mul, 2, name="mul")
        pset.addPrimitive(safe_div, 2, name="div")
        pset.addPrimitive(safe_neg, 1, name="neg")
        pset.addPrimitive(safe_abs, 1, name="abs")
        pset.addPrimitive(safe_sqrt, 1, name="sqrt")
        pset.addPrimitive(safe_log, 1, name="log")
        # avoid lambda for pickling
        pset.addEphemeralConstant("rand", functools.partial(np.random.uniform, -1, 1))

        for i in range(self.n_features):
            pset.renameArguments(**{f"ARG{i}": f"x{i}"})
        return pset

    def setup(self, X: np.ndarray, y: np.ndarray):
        self.n_features = X.shape[1]
        self.X_train = torch.tensor(X, dtype=torch.float32, device=device)
        self.y_train = torch.tensor(y, dtype=torch.float32, device=device)
        self.pset = self._build_primitives()
        self._setup_toolbox()

    def _setup_toolbox(self):
        if "FitnessMin" in creator.__dict__:
            del creator.FitnessMin
        if "Individual" in creator.__dict__:
            del creator.Individual

        # We will minimize MSE, so use negative weight
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=6)
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.expr
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.eval_regression)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register(
            "mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset
        )
        self.toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10)
        )
        self.toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10)
        )

    def eval_regression(self, individual: PrimitiveTree) -> Tuple[float,]:
        """Return MSE to be minimized. Large positive penalty for invalid outputs."""
        try:
            out, node_outputs = quick_evaluate(individual, self.pset, self.X_train)

            # Ensure tensor shape and clamp to avoid overflow
            if not isinstance(out, torch.Tensor):
                return (1e6,)
            out = out.view(-1)
            out = torch.clamp(out, -1e6, 1e6)

            # Check for nan/inf in predictions
            if torch.isnan(out).any() or torch.isinf(out).any():
                return (1e6,)  # Large penalty for invalid predictions

            y_true = self.y_train.view(-1)

            # Fit ridge regressor to predict y from GP predictions
            out_np = out.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()
            ridge = Ridge(alpha=1.0)
            ridge.fit(out_np.reshape(-1, 1), y_true_np)
            # Extract coefficients as floats
            ridge_coef = float(ridge.coef_[0])
            ridge_intercept = float(ridge.intercept_)
            combined = ridge_coef * out + ridge_intercept

            # MSE on combined predictions
            loss = torch.mean((y_true - combined) ** 2)
            val = loss.item()
            if not np.isfinite(val):
                return (1e6,)

            # Now compute gradients
            # First, compute grads_map
            items = [
                (idx, t)
                for idx, t in node_outputs.items()
                if isinstance(t, torch.Tensor)
                and (t.requires_grad or t.grad_fn is not None)
            ]
            if items:
                tensors = [t for _, t in items]
                try:
                    grads = torch.autograd.grad(
                        loss, tensors, retain_graph=True, allow_unused=True
                    )
                    grads_map = {
                        idx: g for (idx, _), g in zip(items, grads) if g is not None
                    }
                except RuntimeError:
                    grads_map = {}
            else:
                grads_map = {}

            # Then compute grad_output
            # Use the last node index as the final output node (root)
            final_idx = max(node_outputs.keys()) if node_outputs else None
            grad_output = grads_map.get(final_idx)
            if grad_output is None:
                try:
                    grad_output = torch.autograd.grad(
                        loss,
                        out,
                        retain_graph=False,
                        allow_unused=True,
                        create_graph=False,
                    )[0]
                    if grad_output is None:
                        grad_output = torch.zeros_like(out)
                except RuntimeError:
                    grad_output = torch.zeros_like(out)

            # Convert maps to numpy (lightweight semantics + gradients)
            grads_map_np: Dict[int, np.ndarray] = {
                idx: g.detach().cpu().float().view(-1).numpy()
                for idx, g in grads_map.items()
            }
            semantics_map_np: Dict[int, np.ndarray] = {
                idx: t.detach().cpu().float().view(-1).numpy()
                for idx, t in node_outputs.items()
                if isinstance(t, torch.Tensor)
            }
            grad_output_np: np.ndarray = (
                grad_output.detach().cpu().float().view(-1).numpy()
            )

            # Strictly control what is stored on the individual to avoid deepcopy issues
            safe_payload = {
                "fitness": getattr(individual, "fitness", None),
                "grads_map": grads_map_np,
                "grad_output": grad_output_np,
                "semantics_map": semantics_map_np,
                "ridge_coef": ridge_coef,
                "ridge_intercept": ridge_intercept,
            }
            individual.__dict__.clear()
            individual.__dict__.update(safe_payload)

            return (val,)
        except Exception:
            return (1e6,)  # Large penalty instead of -inf


def quick_evaluate(
    expr: PrimitiveTree, pset: gp.PrimitiveSet, data: torch.Tensor
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Manual evaluation with per-node tensors (autograd-enabled where applicable).
    Returns (final_output_tensor, dict: node_index -> output_tensor).
    Recognizes both 'ARGi' and 'xi' argument names.

    Important: Ephemeral constants are frozen deterministically on first use to
    avoid re-sampling mismatches between training fitness and later evaluation.
    """
    stack: List[Tuple[object, List[torch.Tensor], int]] = []
    node_outputs: Dict[int, torch.Tensor] = {}
    result: Optional[torch.Tensor] = None

    for idx, node in enumerate(expr):
        stack.append((node, [], idx))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args, node_idx = stack.pop()
            if isinstance(prim, Primitive):
                result = pset.context[prim.name](*args)
            elif isinstance(prim, Terminal):
                name = prim.name
                if name.startswith("ARG") or name.startswith("x"):
                    # Extract column index from name suffix
                    try:
                        col = int(name[3:]) if name.startswith("ARG") else int(name[1:])
                    except ValueError:
                        col = 0
                    result = data[:, col].clone().detach().requires_grad_(True)
                else:
                    # Freeze ephemeral constants deterministically
                    val = getattr(prim, "value", None)
                    # If value is a generator name in the context, sample ONCE and store
                    if isinstance(val, str):
                        gen = pset.context.get(val, None)
                        if callable(gen):
                            try:
                                const = float(gen())
                            except Exception:
                                const = 0.0
                            prim.value = const
                            val = const
                    # If value is a callable (generator), sample ONCE and store
                    elif callable(val):
                        try:
                            const = float(val())
                        except Exception:
                            const = 0.0
                        prim.value = const
                        val = const
                    # Now convert to tensor
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val).float()
                    if isinstance(val, torch.Tensor):
                        result = val.to(device).float()
                    else:
                        try:
                            fv = float(val)
                        except Exception:
                            fv = 0.0
                        result = torch.tensor(fv, dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Unknown node type: {type(prim)}")

            if isinstance(result, torch.Tensor):
                node_outputs[node_idx] = result
            if len(stack) == 0:
                break
            stack[-1][1].append(result)

    if result is None:
        raise RuntimeError("quick_evaluate produced no tensor output")
    return result, node_outputs


def extract_subtrees(individual: PrimitiveTree) -> List[int]:
    subtrees = []
    for i, _ in enumerate(individual):
        subtrees.append(i)
    return subtrees


def _compute_required_semantics(
    current_sem: np.ndarray, grad_vec: np.ndarray
) -> np.ndarray:
    """Heuristic required semantics for a node: take a small step opposite the gradient.
    Scales the gradient to the magnitude of the current semantics to avoid exploding steps.
    """
    try:
        cur = np.asarray(current_sem, dtype=np.float32).reshape(-1)
        g = np.asarray(grad_vec, dtype=np.float32).reshape(-1)
        if cur.size == 0 or g.size == 0 or cur.shape != g.shape:
            return cur
        g_norm = np.linalg.norm(g) + 1e-8
        cur_norm = np.linalg.norm(cur) + 1e-8
        # step size proportional to semantic norm; gamma is a small fraction
        gamma = 0.5
        eta = gamma * (cur_norm / g_norm)
        req = cur - eta * g
        # Clip to reasonable range to avoid numerical issues
        return np.clip(req, -1e6, 1e6)
    except Exception:
        return np.asarray(current_sem, dtype=np.float32).reshape(-1)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.shape != b.shape or a.size == 0:
            return float("inf")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        cos_sim = np.dot(a, b) / (norm_a * norm_b)
        return 1.0 - cos_sim
    except Exception:
        return float("inf")


def swap_subtrees(ind1: PrimitiveTree, idx1: int, ind2: PrimitiveTree, idx2: int):
    """Swap subtrees in-place between two individuals by root indices.
    Save copies of both subtrees first to avoid any override side-effects.
    """
    slice1 = ind1.searchSubtree(idx1)
    slice2 = ind2.searchSubtree(idx2)
    # Make explicit copies of both subtrees before any assignment
    subtree1 = list(ind1[slice1])
    subtree2 = list(ind2[slice2])
    # Replace
    ind1[slice1] = subtree2
    ind2[slice2] = subtree1
    return ind1, ind2


def replace_subtree(
    target: PrimitiveTree, target_idx: int, source: PrimitiveTree, source_idx: int
) -> PrimitiveTree:
    """Replace the subtree at target_idx in target with a copy of the subtree at source_idx from source.
    Does not modify the source individual.
    """
    tgt_slice = target.searchSubtree(target_idx)
    src_slice = source.searchSubtree(source_idx)
    src_copy = list(source[src_slice])
    target[tgt_slice] = src_copy
    return target


def gradient_aligned_crossover(
    p1: PrimitiveTree,
    p2: PrimitiveTree,
):
    """Two one-way replacements (no swap):
    1) Pick s1 in p1 (sem+grad), compute required semantics; find best_s2 in p2 (sem) and copy it to replace s1 in p1.
    2) Pick random s2 in p2 (sem+grad), compute required semantics; find best_s1 in p1 (sem) and copy it to replace s2 in p2.
    If any required info is missing or incompatible, fall back to one-point crossover.
    """
    try:
        grads_map1 = getattr(p1, "grads_map", None) or {}
        sem_map1 = getattr(p1, "semantics_map", None) or {}
        grads_map2 = getattr(p2, "grads_map", None) or {}
        sem_map2 = getattr(p2, "semantics_map", None) or {}
        if not grads_map1 or not sem_map1 or not grads_map2 or not sem_map2:
            return gp.cxOnePoint(p1, p2)

        s1_list = extract_subtrees(p1)
        s2_list = extract_subtrees(p2)
        if not s1_list or not s2_list:
            return gp.cxOnePoint(p1, p2)

        # Candidates
        s1_candidates = [
            i
            for i in s1_list
            if (i in sem_map1)
            and (i in grads_map1)
            and np.asarray(sem_map1[i]).size > 0
            and np.asarray(grads_map1[i]).size > 0
            and np.asarray(sem_map1[i]).shape == np.asarray(grads_map1[i]).shape
        ]
        s2_candidates_sem = [
            j for j in s2_list if (j in sem_map2) and np.asarray(sem_map2[j]).size > 0
        ]
        s2_candidates_grad = [
            j
            for j in s2_list
            if (j in sem_map2)
            and (j in grads_map2)
            and np.asarray(sem_map2[j]).size > 0
            and np.asarray(grads_map2[j]).size > 0
            and np.asarray(sem_map2[j]).shape == np.asarray(grads_map2[j]).shape
        ]
        if not s1_candidates or not s2_candidates_sem or not s2_candidates_grad:
            return gp.cxOnePoint(p1, p2)

        # Step 1: best s2 -> replace s1 in p1
        s1 = int(np.random.choice(s1_candidates))
        sem1 = np.asarray(sem_map1[s1]).reshape(-1)
        g1 = np.asarray(grads_map1[s1]).reshape(-1)
        req1 = _compute_required_semantics(sem1, g1)

        best_s2 = None
        best_score_12 = float("inf")
        for s2 in s2_candidates_sem:
            sem2 = np.asarray(sem_map2[s2]).reshape(-1)
            if sem2.shape != req1.shape:
                continue
            score = _cosine_distance(req1, sem2)
            if score < best_score_12:
                best_score_12 = score
                best_s2 = s2

        if best_s2 is None or not np.isfinite(best_score_12):
            return gp.cxOnePoint(p1, p2)

        # Step 2: best s1 -> replace random s2 in p2
        s2_rand = int(np.random.choice(s2_candidates_grad))
        sem2r = np.asarray(sem_map2[s2_rand]).reshape(-1)
        g2r = np.asarray(grads_map2[s2_rand]).reshape(-1)
        req2 = _compute_required_semantics(sem2r, g2r)

        best_s1 = None
        best_score_21 = float("inf")
        for s1c in s1_list:
            if s1c not in sem_map1:
                continue
            sem1c = np.asarray(sem_map1[s1c]).reshape(-1)
            if sem1c.shape != req2.shape:
                continue
            score = _cosine_distance(req2, sem1c)
            if score < best_score_21:
                best_score_21 = score
                best_s1 = s1c

        if best_s1 is None or not np.isfinite(best_score_21):
            return gp.cxOnePoint(p1, p2)

        # Apply both replacements with pre-copied source subtrees to avoid overlap issues
        p1_tgt_slice = p1.searchSubtree(s1)
        p2_tgt_slice = p2.searchSubtree(s2_rand)
        p2_src_slice = p2.searchSubtree(best_s2)
        p1_src_slice = p1.searchSubtree(best_s1)
        src_best_s2 = list(p2[p2_src_slice])
        src_best_s1 = list(p1[p1_src_slice])
        # Replace: best_s2 -> p1@s1, best_s1 -> p2@s2_rand
        p1[p1_tgt_slice] = src_best_s2
        p2[p2_tgt_slice] = src_best_s1
        return p1, p2
    except Exception:
        return gp.cxOnePoint(p1, p2)


class SafeHallOfFame:
    """HallOfFame replacement that avoids deepcopy of individuals containing torch graphs.
    Stores a sanitized clone with only list structure and fitness values.
    """

    def __init__(self, maxsize: int = 1):
        self.maxsize = maxsize
        self.items: List[PrimitiveTree] = []

    def update(self, population: List[PrimitiveTree]):
        if not population:
            return
        # Select the best individual according to fitness
        best = tools.selBest(population, k=1)[0]
        # Create a shallow structural clone (list copy)
        clone = type(best)(best)
        # Recreate a bare fitness object and copy values
        fit = getattr(best, "fitness", None)
        if fit is not None:
            FClass = fit.__class__
            new_fit = FClass()
            try:
                new_fit.values = tuple(fit.values)
            except Exception:
                pass
            clone.fitness = new_fit
        # Also preserve simple scalar attributes needed for post-hoc evaluation (e.g., ridge correction)
        for attr in ("ridge_coef", "ridge_intercept"):
            if hasattr(best, attr):
                try:
                    setattr(clone, attr, float(getattr(best, attr)))
                except Exception:
                    pass
        # Replace stored items
        self.items = [clone]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PrimitiveTree:
        return self.items[idx]


def run_comparison(n_gen: int = 20, pop_size: int = 50):
    """Run comparison between normal and gradient-aligned crossover"""
    X, y = load_diabetes(return_X_y=True)
    # X, y = make_regression()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    X_train_torch = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32, device=device)

    results = {}

    for crossover_type in ["normal", "gradient_aligned"]:
        wrapper = SklearnRegressionWrapper(n_features=X_train.shape[1])
        wrapper.setup(X_train, y_train)

        population = wrapper.toolbox.population(n=pop_size)
        # Use SafeHallOfFame to avoid deepcopy of individuals with torch data
        hof = SafeHallOfFame(1)

        if crossover_type == "gradient_aligned":
            wrapper.toolbox.register(
                "mate",
                lambda a, b: gradient_aligned_crossover(a, b),
            )
            wrapper.toolbox.decorate(
                "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10)
            )

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(lambda ind: len(ind))
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        start = time.time()
        print(f"Starting {crossover_type} evolution...")
        pop, log = algorithms.eaSimple(
            population,
            wrapper.toolbox,
            cxpb=0.9,
            mutpb=0.1,
            ngen=n_gen,
            stats=mstats,
            halloffame=hof,
            verbose=True,
        )
        elapsed = time.time() - start

        # Compute true R² on train and test for the best individual
        best = hof[0]
        with torch.no_grad():
            # Evaluate using the same interpreter as fitness to avoid any mismatch
            pred_train, _ = quick_evaluate(best, wrapper.pset, X_train_torch)
            pred_test, _ = quick_evaluate(best, wrapper.pset, X_test_torch)
            pred_train = pred_train.view(-1)
            pred_test = pred_test.view(-1)
            # Clamp to be safe
            pred_train = torch.clamp(pred_train, -1e6, 1e6)
            pred_test = torch.clamp(pred_test, -1e6, 1e6)

            # Use the saved ridge coefficients from the best individual to boost performance
            if hasattr(best, "ridge_coef") and hasattr(best, "ridge_intercept"):
                ridge_coef = float(getattr(best, "ridge_coef"))
                ridge_intercept = float(getattr(best, "ridge_intercept"))
            else:
                # Fallback: fit a small ridge on-the-fly to align with training-time fitness
                from sklearn.linear_model import Ridge as _Ridge

                _ridge = _Ridge(alpha=1.0)
                _ridge.fit(
                    pred_train.detach().cpu().view(-1, 1).numpy(),
                    y_train_torch.detach().cpu().numpy(),
                )
                ridge_coef = float(_ridge.coef_[0])
                ridge_intercept = float(_ridge.intercept_)

            # Combine GP and ridge predictions
            pred_train = ridge_coef * pred_train + ridge_intercept
            pred_test = ridge_coef * pred_test + ridge_intercept
            # Clamp again to be safe
            pred_train = torch.clamp(pred_train, -1e6, 1e6)
            pred_test = torch.clamp(pred_test, -1e6, 1e6)

            def r2(y_true, y_pred):
                y_true = y_true.view(-1)
                y_pred = y_pred.view(-1)
                ss_res = torch.sum((y_true - y_pred) ** 2)
                ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
                return (1.0 - (ss_res / (ss_tot + 1e-12))).item()

            r2_train = r2(y_train_torch, pred_train)
            r2_test = r2(y_test_torch, pred_test)

        results[crossover_type] = {
            "best": best,
            "best_fitness_raw": hof[0].fitness.values[0],  # MSE
            "log": log,
            "time": elapsed,
            "r2_train": r2_train,
            "r2_test": r2_test,
        }
        print(
            f"{crossover_type:20s}: best_MSE={results[crossover_type]['best_fitness_raw']:.6f}, "
            f"R²_train={r2_train:.4f}, R²_test={r2_test:.4f}, time={elapsed:.2f}s"
        )

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Gradient Crossover Comparison")
    print("=" * 60)
    results = run_comparison(n_gen=100, pop_size=100)
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
