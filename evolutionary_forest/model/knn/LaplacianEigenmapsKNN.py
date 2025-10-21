import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances

from evolutionary_forest.model.OptimalKNN import OptimalKNN


class LaplacianEigenmapsKNN(OptimalKNN):
    """
    A simplified KNN model that learns a parametric Laplacian Eigenmaps embedding
    from label similarity, then runs KNN in that learned manifold space.

    Inherits from OptimalKNN for consistency with API (fit, transform, predict),
    but overrides the core training routine to:
        1. Compute label-based similarity S and Laplacian L
        2. Solve (X^T L X) w = λ (X^T D X) w
        3. Use eigenvectors to define transformation W
        4. Fit KNN on XW

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors for KNN.
    n_components : int or None
        Dimensionality of the embedding space. Defaults to p-1.
    sigma : float
        Bandwidth of the label similarity RBF kernel.
    distance : str
        'Uniform' or 'Euclidean' to set KNN weighting.
    ridge : float
        Regularization term added to X^T D X for numerical stability.
    """

    def __init__(
        self,
        n_neighbors=5,
        n_components=None,
        sigma=1.0,
        distance="Uniform",
        ridge=1e-8,
        random_seed=0,
        base_learner=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            distance=distance,
            random_seed=random_seed,
            n_groups=1,
            base_learner=base_learner,
        )
        self.n_components = n_components
        self.sigma = sigma
        self.ridge = ridge

        # learned params
        self.W_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        self.n_features_in_ = p

        # --- Step 1: Label-based similarity and Laplacian ---
        y = y.reshape(-1, 1)
        y_dist = pairwise_distances(y, metric="euclidean")
        S = np.exp(-(y_dist**2) / (2 * self.sigma**2))
        Dmat = np.diag(S.sum(axis=1))
        L = Dmat - S

        # --- Step 2: Generalized eigenproblem ---
        A = X.T @ L @ X
        B = X.T @ Dmat @ X + self.ridge * np.eye(p)

        n_comp = self.n_components or max(1, p - 1)
        vals, vecs = eigh(A, B)
        idx = np.argsort(vals)  # ascending
        vals, vecs = vals[idx], vecs[:, idx]

        # Skip the trivial eigenvector if eigenvalue ≈ 0
        start = 1 if vals[0] < 1e-12 and p > 1 else 0
        end = min(start + n_comp, p)
        self.W_ = vecs[:, start:end]

        # --- Step 3: Transform and fit KNN ---
        transformed_X = X @ self.W_
        self.training_data = transformed_X
        self.get_knn_model(self.base_learner, self.distance)
        self.knn.fit(transformed_X, y)

        return self

    def transform(self, X):
        if self.W_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return np.asarray(X) @ self.W_

    def predict(self, X, return_transformed=False):
        test_data = self.transform(X)
        prediction = self.knn.predict(test_data)
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        if return_transformed:
            return prediction, test_data
        return prediction


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.neighbors import KNeighborsRegressor
    import numpy as np

    # Load dataset
    X, y = load_diabetes(return_X_y=True)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # ---- Standard KNN baseline ----
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    r2_knn = r2_score(y_test, y_pred_knn)

    # ---- LaplacianEigenmapsKNN ----
    lem_knn = LaplacianEigenmapsKNN(n_neighbors=5)
    lem_knn.fit(X_train, y_train)
    y_pred_lem = lem_knn.predict(X_test)
    r2_lem = r2_score(y_test, y_pred_lem)

    # ---- Print comparison ----
    print("Test R² (Standard KNN):          ", round(r2_knn, 4))
    print("Test R² (LaplacianEigenmapsKNN): ", round(r2_lem, 4))
