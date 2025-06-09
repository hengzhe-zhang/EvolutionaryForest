import pickle
import random
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize
from deap import base, creator, tools, gp
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn

from evolutionary_forest.component.semantic_library.surrogate_tool import (
    remap_gp_tree_variables,
)


class SplitRidge:
    def __init__(self, alpha=1e-6, batch_size=50000):
        self.alpha = alpha
        self.batch_size = batch_size
        self.ridges = []
        self.n_targets = 0

    def fit(self, X, Y):
        n_targets = Y.shape[1]
        self.n_targets = n_targets
        self.ridges = []
        for start in range(0, n_targets, self.batch_size):
            end = min(start + self.batch_size, n_targets)
            ridge = Ridge(alpha=self.alpha, fit_intercept=False)
            ridge.fit(X, Y[:, start:end])
            self.ridges.append(ridge)
        return self

    def predict(self, X):
        preds = [ridge.predict(X) for ridge in self.ridges]
        return np.hstack(preds)


class FastFeatureAttentionRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Linear(input_dim, input_dim)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        attn_weights = torch.softmax(self.attn(x), dim=1)  # (batch, input_dim)
        x_attn = x * attn_weights
        x_hidden = self.relu(self.hidden(x_attn))
        return self.out(x_hidden)


class EnsembleRidge:
    def __init__(self, alpha=1e-6):
        self.alpha = alpha
        self.models = []

    def fit(self, X_list, Y_list):
        assert len(X_list) == len(Y_list)
        self.models = []
        for X, Y in zip(X_list, Y_list):
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(X, Y)
            self.models.append(model)

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        preds = np.stack(preds, axis=0)
        return np.mean(preds, axis=0)


def center_and_normalize(X, axis=0):
    mean = np.mean(X, axis=axis, keepdims=True)
    Xc = X - mean
    return normalize(Xc, axis=axis)


def protected_div(x, y):
    return np.where(np.abs(y) > 1e-8, np.true_divide(x, y), 1.0)


def protected_log(x):
    return np.log(np.abs(x) + 1)


def protected_sqrt(x):
    return np.sqrt(np.abs(x))


def sin_pi(x):
    return np.sin(np.pi * x)


def cos_pi(x):
    return np.cos(np.pi * x)


class SemanticLibraryRegressor:
    def __init__(self, n_basis=100, alpha=1e-6, n_equations=1000):
        self.n_basis = n_basis
        self.alpha = alpha
        self.n_equations = n_equations
        self.expressions = []
        self.basis_indices = []
        self.pset = None
        self.ridge = None

        self._last_X = None
        self._last_pred_semantics = None
        self._last_mapping = None

    def _setup_pset(self, n_features):
        pset = gp.PrimitiveSet("MAIN", n_features)
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(sin_pi, 1)
        pset.addPrimitive(cos_pi, 1)
        pset.addPrimitive(np.tanh, 1)
        pset.addPrimitive(np.square, 1)
        pset.addPrimitive(protected_log, 1)
        pset.addPrimitive(protected_sqrt, 1)
        pset.addPrimitive(np.minimum, 2)
        pset.addPrimitive(np.maximum, 2)
        for i in range(n_features):
            pset.renameArguments(**{f"ARG{i}": f"x{i}"})
        return pset

    def _generate_expressions(self, n_features):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        pset = self._setup_pset(n_features)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genGrow, pset=pset, min_=1, max_=3)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.expr
        )
        exprs = set()
        out = []
        attempts = 0
        while len(out) < self.n_equations and attempts < 10 * self.n_equations:
            expr = toolbox.individual()
            s = str(expr)
            if s not in exprs:
                exprs.add(s)
                out.append(expr)
            attempts += 1
        return out, pset

    def _compile_expr(self, expr):
        func = gp.compile(expr, self.pset)

        def f(X):
            args = [X[:, i] for i in range(X.shape[1])]
            return func(*args)

        return f

    def fit(self, X):
        n_features = X.shape[1]
        self.expressions, self.pset = self._generate_expressions(n_features)
        n_expressions = len(self.expressions)
        n_samples = X.shape[0]
        semantics = np.zeros((n_samples, n_expressions))
        for i, expr in enumerate(self.expressions):
            f = self._compile_expr(expr)
            semantics[:, i] = f(X)
        semantics = center_and_normalize(semantics, axis=0)

        # Deduplicate semantics and expressions
        arr = np.round(semantics, 8)  # 8 decimals should be robust, change as needed
        arr_view = arr.T.copy().view([("", arr.dtype)] * arr.shape[0]).squeeze()
        _, unique_idx = np.unique(arr_view, return_index=True)
        unique_idx = np.sort(unique_idx)
        semantics = semantics[:, unique_idx]
        self.expressions = [self.expressions[i] for i in unique_idx]

        # Now sample bases from the unique set
        n_unique = semantics.shape[1]
        self.basis_indices = random.sample(range(n_unique), min(self.n_basis, n_unique))
        X_basis = semantics[:, self.basis_indices]

        self.ridge = SplitRidge(alpha=self.alpha)
        self.ridge.fit(X_basis, semantics)

    def predict(self, X, return_mapping=False):
        target_n_features = len(self.pset.arguments)
        if X.shape[1] < target_n_features:
            X, mapping = self._pad_features(X, target_n_features)
        else:
            mapping = list(range(X.shape[1]))
        semantics_basis = np.zeros((X.shape[0], len(self.basis_indices)))
        for i, idx in enumerate(self.basis_indices):
            semantics_basis[:, i] = self._compile_expr(self.expressions[idx])(X)
        semantics_basis = center_and_normalize(semantics_basis, axis=0)
        result = center_and_normalize(self.ridge.predict(semantics_basis), axis=0)
        if return_mapping:
            return result, mapping
        else:
            return result

    def _pad_features(self, X, target_n_features, seed=0):
        rng = np.random.RandomState(seed)
        n_samples, n_features = X.shape
        pad_indices = rng.choice(
            n_features, target_n_features - n_features, replace=True
        )
        X_pad = X[:, pad_indices]
        X_new = np.concatenate([X, X_pad], axis=1)
        # Mapping: original features [0,1,...,n_features-1] + padded ones
        mapping = list(range(n_features)) + pad_indices.tolist()
        return X_new, mapping

    def retrieve_nearest(self, X, vec, topk=1):
        if X is self._last_X:
            pred_semantics = self._last_pred_semantics
            mapping = self._last_mapping
        else:
            pred_semantics, mapping = self.predict(X, return_mapping=True)
            self._last_X = X
            self._last_pred_semantics = pred_semantics
            self._last_mapping = mapping

        vec = np.asarray(vec).reshape(1, -1)
        sims = cosine_similarity(pred_semantics.T, vec).flatten()
        topk_idx = np.argsort(sims)[-topk:][::-1]
        results = []
        for idx in topk_idx:
            expr = self.expressions[idx]
            expr_remapped = remap_gp_tree_variables(expr, mapping, self.pset)
            results.append((expr_remapped, idx, sims[idx]))
        return results  # List of (remapped_expr, idx, similarity)

    def save(self, filepath):
        """Save the regressor's state to a file."""
        state = {
            "n_basis": self.n_basis,
            "alpha": self.alpha,
            "n_equations": self.n_equations,
            "expressions": self.expressions,
            "basis_indices": self.basis_indices,
            "pset": self.pset,
            "ridge": self.ridge,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath):
        """Load a regressor's state from a file and return the object."""
        # Make sure DEAP creator classes are registered
        from deap import creator, base, gp

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

        with open(filepath, "rb") as f:
            state = pickle.load(f)
        obj = cls(
            n_basis=state["n_basis"],
            alpha=state["alpha"],
            n_equations=state["n_equations"],
        )
        obj.expressions = state["expressions"]
        obj.basis_indices = state["basis_indices"]
        obj.pset = state["pset"]
        obj.ridge = state["ridge"]
        return obj


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    n_train = 500
    n_test = 50
    n_basis = 100
    n_features = 10

    X_train = np.random.uniform(-10, 10, size=(n_train, n_features))
    X_test = np.random.uniform(-10, 10, size=(n_test, 3))

    print("Fitting semantic library regressor...")
    reg = SemanticLibraryRegressor(n_basis=n_basis, alpha=1e-6, n_equations=500000)
    reg.fit(X_train)

    reg.save("result/fast_semantic_library.pkl")
    reg = SemanticLibraryRegressor.load("result/fast_semantic_library.pkl")

    # --- Predict semantics (regressor handles padding internally) ---
    print("\n=== Test: All features available (with auto-padding) ===")
    start = time.time()
    semantics_pred = reg.predict(X_test)
    print(f"Prediction time: {time.time() - start:.3f}s, shape: {semantics_pred.shape}")

    # --- Top-k retrieval test ---
    case = np.sin(X_test[:, 0]) + X_test[:, 1] + X_test[:, 2]
    print("\n=== Top-10 semantic retrieval ===")
    for expr, idx, sim in reg.retrieve_nearest(X_test, case, topk=10):
        print(f"Index: {idx}, Similarity: {sim:.4f}, Expr: {expr}")

    # --- R² / cosine similarity evaluation ---
    print("\n=== Cosine Similarity Evaluation on Test Set ===")

    # Compute true semantics
    def get_true_semantics(reg, X):
        semantics = np.zeros((X.shape[0], len(reg.expressions)))
        for i, expr in enumerate(reg.expressions):
            semantics[:, i] = reg._compile_expr(expr)(X)
        return center_and_normalize(semantics, axis=0)

    start = time.time()
    X_test_padded, _ = reg._pad_features(X_test, len(reg.pset.arguments))
    semantics_true = get_true_semantics(reg, X_test_padded)
    print(f"True semantics computation: {time.time() - start:.3f}s")

    cos_sim = np.sum(
        center_and_normalize(semantics_pred, axis=0) * semantics_true,
        axis=0,
    )
    print(f"Average cosine similarity: {np.mean(cos_sim):.4f}")

    # --- Top-k sample retrieval by predicted semantics ---
    def topk_semantic_search(
        semantics_pred, semantics_true, k=10, query_range=(1000, 1010)
    ):
        print("\n=== Top-k Closest Sample Search in Predicted Semantics ===")
        for query_idx in range(*query_range):
            query_vec = semantics_true[:, query_idx].reshape(1, -1)
            sim_pred = cosine_similarity(query_vec, semantics_pred.T).flatten()
            topk_idx = np.argsort(sim_pred)[-k:][::-1]
            topk_true_sem = semantics_true[:, topk_idx].T
            sim_to_query = cosine_similarity(query_vec, topk_true_sem).flatten()
            best_in_topk = np.argmax(sim_to_query)
            print(f"\nQuery: {query_idx}")
            print(f"Top-{k} predicted: {topk_idx.tolist()}")
            print(
                f"Best true in top-{k}: {topk_idx[best_in_topk]} (sim: {sim_to_query[best_in_topk]:.4f})"
            )
            print(f"True sims in top-{k}: {sim_to_query.round(4).tolist()}")
            print(
                f"Best true overall: {np.argmax(cosine_similarity(query_vec, semantics_true.T))}"
            )

    topk_semantic_search(semantics_pred, semantics_true, k=10, query_range=(1000, 1010))
