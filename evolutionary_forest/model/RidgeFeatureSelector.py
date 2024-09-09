import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV


def feature_selection(ind):
    selector = SequentialFeatureSelector(
        Ridge(),
        n_features_to_select="auto",
        direction="forward",
        scoring="r2",
        cv=5,
    )
    selector.fit(X_train, y_train)
    selected_features_ = list(selector.get_support())
    ind.gene = [ind.gene[idx] for idx in selected_features_]
    return ind


class RidgeForwardFeatureSelector:
    def __init__(self, n_features_to_select="auto", alpha_per_target=None):
        self.n_features_to_select = n_features_to_select
        self.alpha_per_target = alpha_per_target
        self.selected_features_ = None
        self.coef_ = None
        self.model = None

    def fit(self, X_train, y_train):
        # Perform forward feature selection with Ridge regression
        selector = SequentialFeatureSelector(
            Ridge(),
            n_features_to_select=self.n_features_to_select,
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
    from sklearn.datasets import make_regression, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Generate synthetic data
    # X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=0)
    X, y = load_diabetes(return_X_y=True)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Initialize and fit the feature selector
    selector = RidgeForwardFeatureSelector()
    selector.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = selector.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Selected features: {np.where(selector.selected_features_)[0]}")
    print(f"Coefficients: {selector.coef_}")
    print(f"Mean Squared Error on test set: {mse}")
