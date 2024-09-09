from numbers import Integral, Real

import numpy as np
from sklearn.base import is_classifier, clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, check_cv


class SequentialFeatureSelectorNew(SequentialFeatureSelector):
    def fit(self, X, y=None):
        tags = self._get_tags()
        X = self._validate_data(
            X,
            accept_sparse="csc",
            ensure_min_features=2,
            force_all_finite=not tags.get("allow_nan", True),
        )
        n_features = X.shape[1]

        if self.n_features_to_select == "auto":
            if self.tol is not None:
                # With auto feature selection, `n_features_to_select_` will be updated
                # to `support_.sum()` after features are selected.
                self.n_features_to_select_ = n_features
            else:
                self.n_features_to_select_ = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):
            if self.n_features_to_select >= n_features:
                raise ValueError("n_features_to_select must be < n_features.")
            self.n_features_to_select_ = self.n_features_to_select
        elif isinstance(self.n_features_to_select, Real):
            self.n_features_to_select_ = int(n_features * self.n_features_to_select)

        if self.tol is not None and self.tol < 0 and self.direction == "forward":
            raise ValueError("tol must be positive when doing forward selection")

        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        cloned_estimator = clone(self.estimator)

        # the current mask corresponds to the set of features:
        # - that we have already *selected* if we do forward selection
        # - that we have already *excluded* if we do backward selection
        current_mask = np.zeros(shape=n_features, dtype=bool)
        n_iterations = (
            self.n_features_to_select_
            if self.n_features_to_select == "auto" or self.direction == "forward"
            else n_features - self.n_features_to_select_
        )

        old_score = -np.inf
        is_auto_select = self.tol is not None and self.n_features_to_select == "auto"
        for _ in range(n_iterations):
            new_feature_idx, new_score = self._get_best_new_feature_score(
                cloned_estimator, X, y, cv, current_mask
            )
            if is_auto_select and ((new_score - old_score) < self.tol):
                break

            old_score = new_score
            current_mask[new_feature_idx] = True

        if self.direction == "backward":
            current_mask = ~current_mask

        self.support_ = current_mask
        self.n_features_to_select_ = self.support_.sum()

        return self


def feature_selection(ind, X_train, y_train):
    if X_train.shape[1] == 1:
        return ind, X_train
    selector = SequentialFeatureSelectorNew(
        Ridge(),
        n_features_to_select="auto",
        direction="forward",
        tol=0,
        scoring="r2",
        cv=5,
    )
    selector.fit(X_train, y_train)
    selected_features_ = selector.get_support()
    ind.gene = [
        ind.gene[idx] for idx in range(len(ind.gene)) if selected_features_[idx]
    ]
    X_train = X_train[:, selected_features_]
    return ind, X_train


class RidgeForwardFeatureSelector:
    def __init__(self, n_features_to_select="auto", alpha_per_target=None):
        self.n_features_to_select = n_features_to_select
        self.alpha_per_target = alpha_per_target
        self.selected_features_ = None
        self.coef_ = None
        self.model = None

    def fit(self, X_train, y_train):
        # Perform forward feature selection with Ridge regression
        selector = SequentialFeatureSelectorNew(
            Ridge(),
            n_features_to_select=self.n_features_to_select,
            tol=0,
            direction="forward",
            scoring="r2",
            cv=5,
        )
        selector.fit(X_train, y_train)
        self.selected_features_ = selector.get_support()

        # Fit RidgeCV model using selected features
        ridge = (
            RidgeCV(alphas=self.alpha_per_target)
            if self.alpha_per_target
            else RidgeCV()
        )
        self.model = ridge.fit(X_train[:, self.selected_features_], y_train)

        # Store the coefficients
        self.coef_ = np.zeros(X_train.shape[1])
        self.coef_[self.selected_features_] = self.model.coef_

    def predict(self, X):
        if self.model is None or self.selected_features_ is None:
            raise RuntimeError("You must fit the model before predicting.")
        X_transformed = X[:, self.selected_features_]
        return self.model.predict(X_transformed)


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error

    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=0)
    # X, y = load_diabetes(return_X_y=True)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ridge = RidgeCV()
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test set: {mse}")

    for _ in range(5):
        # Initialize and fit the feature selector
        selector = RidgeForwardFeatureSelector()
        selector.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = selector.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        X_train = X_train[:, selector.selected_features_]
        X_test = X_test[:, selector.selected_features_]

        print(f"Selected features: {np.where(selector.selected_features_)[0]}")
        print(f"Coefficients: {selector.coef_}")
        print(f"Mean Squared Error on test set: {mse}")
