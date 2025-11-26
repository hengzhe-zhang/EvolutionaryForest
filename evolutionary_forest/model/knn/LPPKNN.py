import numpy as np
from scipy.linalg import eigh

from evolutionary_forest.component.ensemble_selection.dpp_selection import (
    compute_similarity,
)
from evolutionary_forest.model.OptimalKNN import OptimalKNN


class LPPKNN(OptimalKNN):
    """
    Parametric Laplacian Eigenmaps + KNN
    Learns a projection W via label-similarity Laplacian, then runs KNN in embedded space.
    """

    def __init__(
        self,
        n_neighbors=5,
        n_components=None,
        gamma=None,
        distance="Uniform",
        ridge=1e-8,
        knn_subsampling=100,
        random_seed=0,
        base_learner=None,
        **params,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            distance=distance,
            random_seed=random_seed,
            base_learner=base_learner,
        )
        self.n_components = n_components
        self.gamma = gamma
        self.ridge = ridge
        self.knn_subsampling = knn_subsampling
        self.W_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n, p = X.shape
        self.n_features_in_ = p

        # ---- Step 0: Optional subsampling ----
        if len(y) > self.knn_subsampling:
            subsample_indices = np.random.choice(
                len(y), self.knn_subsampling, replace=False
            )
            X_sub = X[subsample_indices]
            y_sub = y[subsample_indices]
        else:
            X_sub, y_sub = X, y

        # ---- Step 2: Label-based similarity and Laplacian ----
        y_sub = y_sub.reshape(-1, 1)
        S = compute_similarity(y_sub, metric="rbf", gamma=self.gamma)

        Dmat = np.diag(S.sum(axis=1))
        L = Dmat - S

        # ---- Step 3: Generalized eigenproblem ----
        A = X_sub.T @ L @ X_sub
        B = X_sub.T @ Dmat @ X_sub + self.ridge * np.eye(p)

        vals, vecs = eigh(A, B)
        idx = np.argsort(vals)
        vals, vecs = vals[idx], vecs[:, idx]

        # Determine dimensionality (skip trivial eigenvector if necessary)
        n_comp = self.n_components or max(1, p - 1)
        start = 1 if vals[0] < 1e-12 and p > 1 else 0
        end = min(start + n_comp, p)
        self.W_ = vecs[:, start:end]

        # ---- Step 4: Transform full training data & fit KNN ----
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
            return prediction.flatten(), test_data
        return prediction.flatten()

    def get_feature_importance(self, normalize=True):
        """Return feature importance based on the learned projection matrix W_."""
        if self.W_ is None:
            n_features = getattr(self, "n_features_in_", 1)
            return np.ones(n_features) / n_features
        
        # Compute importance as sum of squared weights per input feature
        imp = np.sum(self.W_**2, axis=1)
        
        if normalize:
            imp = imp / (np.sum(imp) + 1e-12)
        return imp


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.neighbors import KNeighborsRegressor

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

    # ---- LPPKNN ----
    lem_knn = LPPKNN(n_neighbors=5)
    lem_knn.fit(X_train, y_train)
    y_pred_lem = lem_knn.predict(X_test)
    r2_lem = r2_score(y_test, y_pred_lem)

    # ---- Print comparison ----
    print("Test R² (Standard KNN):          ", round(r2_knn, 4))
    print("Test R² (LPPKNN): ", round(r2_lem, 4))
