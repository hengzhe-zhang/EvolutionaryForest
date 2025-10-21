from sklearn.cross_decomposition import PLSRegression
import numpy as np

from evolutionary_forest.model.OptimalKNN import OptimalKNN


class PLSKNN(OptimalKNN):
    """
    Simplified OptimalKNN subclass using sklearn's PLSRegression
    for feature transformation.

    - No subsampling, grouping, or Laplacian regularization
    - n_components = X.shape[1] (full dimension)
    - No timing or logging
    """

    def __init__(
        self,
        n_neighbors=5,
        distance="Uniform",
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
        self.pls = None

    def fit(self, X, y):
        # Initialize KNN
        self.get_knn_model(self.base_learner, self.distance)
        self.n_features_in_ = X.shape[1]

        # Use all features as components
        n_comp = X.shape[1]
        self.pls = PLSRegression(n_components=n_comp)
        self.pls.fit(X, y)

        # Transform data and fit KNN
        training_data = self.pls.transform(X)
        self.knn.fit(training_data, y)

        return self

    def transform(self, X):
        if self.pls is None:
            raise ValueError("PLS model not fitted yet. Call fit() first.")
        return self.pls.transform(X)

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

    # ---- PLSKNN (PLS + KNN) ----
    pls_knn = PLSKNN(n_neighbors=5)
    pls_knn.fit(X_train, y_train)
    y_pred_pls = pls_knn.predict(X_test)
    r2_pls = r2_score(y_test, y_pred_pls)

    # ---- Print comparison ----
    print("Test R² (Standard KNN):", round(r2_knn, 4))
    print("Test R² (PLS-KNN):     ", round(r2_pls, 4))
