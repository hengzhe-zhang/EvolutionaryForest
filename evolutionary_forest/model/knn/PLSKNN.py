from sklearn.cross_decomposition import PLSRegression
import numpy as np
from sklearn.feature_selection import VarianceThreshold

from evolutionary_forest.model.OptimalKNN import OptimalKNN


class PLSKNN(OptimalKNN):
    """
    Simplified OptimalKNN subclass using sklearn's PLSRegression
    for feature transformation.

    - Removes constant features before PLS
    - No subsampling, grouping, or Laplacian regularization
    - n_components = X.shape[1] after constant-feature removal
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
        self.var_thresh = None
        self.n_features_in_ = None

    def fit(self, X, y):
        # Remove constant features
        self.var_thresh = VarianceThreshold(threshold=0.0)
        X_filtered = self.var_thresh.fit_transform(X)
        self.n_features_in_ = X_filtered.shape[1]

        # Initialize KNN
        self.get_knn_model(self.base_learner, self.distance)

        # Use all remaining features as PLS components
        n_comp = X_filtered.shape[1]
        self.pls = PLSRegression(n_components=n_comp)

        try:
            self.pls.fit(X_filtered, y)
            training_data = self.pls.transform(X_filtered)
        except ValueError as e:
            print(
                f"[PLSKNN] Warning: PLS failed with error '{e}'. Using identity transform instead."
            )
            self.pls = None
            training_data = X_filtered

        # Fit KNN
        self.knn.fit(training_data, y)

        return self

    def transform(self, X):
        if self.var_thresh is None:
            return X
        X_filtered = self.var_thresh.transform(X)
        if self.pls is None:
            return X_filtered
        return self.pls.transform(X_filtered)

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
