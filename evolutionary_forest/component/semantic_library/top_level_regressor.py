from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_friedman1
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from evolutionary_forest.component.semantic_library.sympy_converter import tree_to_sympy
from evolutionary_forest.component.semantic_library.surrogate import *


from sklearn.linear_model import Ridge

import numpy as np
import copy


class BoostedSemanticRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_path, n_boosting_rounds=10, topk=100, learning_rate=1.0):
        self.model_path = model_path
        self.n_boosting_rounds = n_boosting_rounds
        self.topk = topk
        self.learning_rate = learning_rate
        self.expressions = []
        self.scalers = []
        self.semantics = []  # List of semantic outputs (training y_expr for each expr)
        self.pset = None
        self.semantic_model = None

        # For storing the historically best set:
        self.best_expressions = []
        self.best_scalers = []
        self.best_score = float("-inf")

    def fit(self, X, vec):
        self.semantic_model = SemanticLibraryRegressor.load(self.model_path)
        self.pset = self.semantic_model.pset

        self.expressions = []
        self.scalers = []
        self.semantics = []  # This now stores scaled outputs
        self.best_expressions = []
        self.best_scalers = []
        self.best_score = float("-inf")

        y_pred = np.zeros_like(vec, dtype=np.float64)
        residual = vec - y_pred

        round_count = 0

        while (
            round_count < self.n_boosting_rounds or round_count < self.n_boosting_rounds
        ):
            # Rolling window logic
            if round_count < self.n_boosting_rounds:
                # Normal boosting: just fit to the current residual
                current_residual = residual
            else:
                # Remove the oldest component (first in list)
                del self.expressions[0]
                del self.scalers[0]
                del self.semantics[0]

                # Recompute prediction and residual after removal (using *scaled* semantics)
                y_pred = np.zeros_like(vec, dtype=np.float64)
                for sem in self.semantics:
                    y_pred += self.learning_rate * sem.flatten()
                current_residual = vec - y_pred

            # Retrieve candidates for the current residual
            retrieved = self.semantic_model.retrieve_nearest(
                X, current_residual, topk=self.topk
            )

            best_r2 = float("-inf")
            best_expr = None
            best_scaler = None
            best_semantic = None

            for expr, idx, sim in retrieved:
                func = gp.compile(expr, self.pset)
                n_features = len(self.pset.arguments)
                args = [X[:, i] if i < X.shape[1] else None for i in range(n_features)]
                y_expr = func(*args).reshape(-1, 1)
                scaler = Ridge(alpha=1e-6, fit_intercept=True)
                scaler.fit(y_expr, current_residual)
                y_pred_scaled = scaler.predict(y_expr)
                score = r2_score(current_residual, y_pred_scaled)
                if score > best_r2:
                    best_r2 = score
                    best_expr = expr
                    best_scaler = scaler
                    best_semantic = y_pred_scaled  # store scaled output
            # Add the new best component
            self.expressions.append(best_expr)
            self.scalers.append(best_scaler)
            self.semantics.append(best_semantic)

            # Update y_pred and residual using current ensemble (sum *scaled* semantics)
            y_pred = np.zeros_like(vec, dtype=np.float64)
            for sem in self.semantics:
                y_pred += self.learning_rate * sem.flatten()
            current_r2 = r2_score(vec, y_pred)
            residual = vec - y_pred

            print(
                f"Round {round_count + 1} - expr: {best_expr}, current ensemble train R2: {current_r2:.4f}"
            )

            # Save best historical ensemble (copy all scaled semantics as well)
            if current_r2 > self.best_score:
                self.best_score = current_r2
                self.best_expressions = copy.deepcopy(self.expressions)
                self.best_scalers = copy.deepcopy(self.scalers)

            round_count += 1

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for expr, scaler in zip(self.best_expressions, self.best_scalers):
            func = gp.compile(expr, self.pset)
            n_features = len(self.pset.arguments)
            args = [X[:, i] if i < X.shape[1] else None for i in range(n_features)]
            y_expr = func(*args).reshape(-1, 1)
            y_pred += self.learning_rate * scaler.predict(y_expr)
        return y_pred

    def model(self):
        import sympy

        exprs = self.best_expressions
        scalers = self.best_scalers
        total = 0
        for expr, scaler in zip(exprs, scalers):
            sym = tree_to_sympy(expr, self.pset)
            coef = scaler.coef_.flatten()[0]
            intercept = scaler.intercept_.item()
            total += self.learning_rate * (coef * sym + intercept)
        print("Boosted symbolic model:")
        sympy.pprint(total, use_unicode=True)
        return total


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    # --- Data loading and scaling ---
    # X, y, _ = get_dataset({"dataset": 586})
    X, y = make_friedman1(n_features=10, noise=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    x_scaler = MinMaxScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    # --- TopLevelSemanticRegressor ---
    top_regressor = BoostedSemanticRegressor(
        n_boosting_rounds=1, topk=100, model_path="result/fast_semantic_library.pkl"
    )
    top_regressor.fit(X_train, y_train)
    y_pred_train_sem = top_regressor.predict(X_train)
    y_pred_test_sem = top_regressor.predict(X_test)
    y_pred_test_sem = (
        10 * np.sin(np.pi * X_test[:, 0] * X_test[:, 1])
        + 20 * (X_test[:, 2] - 0.5) ** 2
        + 10 * X_test[:, 3]
        + 5 * X_test[:, 4]
    )
    y_pred_test_sem = y_pred_test_sem.reshape(-1, 1)
    y_pred_test_sem = (
        LinearRegression().fit(y_pred_test_sem, y_test).predict(y_pred_test_sem)
    )

    print("Semantic Regressor:")
    print(f"  R2 on training set: {r2_score(y_train, y_pred_train_sem):.3f}")
    print(f"  R2 on test set:     {r2_score(y_test, y_pred_test_sem):.3f}")

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_train_lr = lr.predict(X_train)
    y_pred_test_lr = lr.predict(X_test)

    print("\nLinear Regression:")
    print(f"  R2 on training set: {r2_score(y_train, y_pred_train_lr):.3f}")
    print(f"  R2 on test set:     {r2_score(y_test, y_pred_test_lr):.3f}")

    # --- Random Forest ---
    rf = RandomForestRegressor(random_state=0)
    rf.fit(X_train, y_train)
    y_pred_train_rf = rf.predict(X_train)
    y_pred_test_rf = rf.predict(X_test)

    print("\nRandom Forest:")
    print(f"  R2 on training set: {r2_score(y_train, y_pred_train_rf):.3f}")
    print(f"  R2 on test set:     {r2_score(y_test, y_pred_test_rf):.3f}")
