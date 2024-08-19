from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FeatureBoundsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_min_ = None
        self.feature_max_ = None

    def fit(self, X, y=None):
        # Calculate the minimum and maximum values for each feature
        self.feature_min_ = np.min(X, axis=0)
        self.feature_max_ = np.max(X, axis=0)
        return self

    def transform(self, X):
        # Clip the features in X to the stored minimum and maximum values
        X_transformed = np.clip(X, self.feature_min_, self.feature_max_)
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=0
    )

    # Initialize and fit the transformer
    transformer = FeatureBoundsTransformer()
    X_train_transformed = transformer.fit_transform(X_train)

    # Transform the test set
    X_test_transformed = transformer.transform(X_test)

    # Print results
    print("Original X_test:")
    print(X_test)
    print("\nTransformed X_test:")
    print(X_test_transformed)
