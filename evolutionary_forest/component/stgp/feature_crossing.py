import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from evolutionary_forest.component.stgp.fast_binary_encoder import BinaryEncoder


class OrdinalEncoder1D(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class FeatureCrossBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mode):
        # Initialize the appropriate encoder based on the specified mode
        self.mode = mode
        if mode not in ["binary", "ordinal"]:
            raise ValueError("Mode must be 'binary' or 'ordinal'")
        if mode == "binary":
            self.encoder = BinaryEncoder()
        elif mode == "ordinal":
            self.encoder = OrdinalEncoder1D()

    def fit(self, X, y=None):
        # Create a combined feature by concatenating all feature columns with an underscore
        combined_features = self._combine_features(X)
        self.encoder.fit(combined_features)
        return self

    def transform(self, X):
        # Recreate the combined feature in the same way as during fit
        combined_features = self._combine_features(X)
        return self.encoder.transform(combined_features)

    def _combine_features(self, X):
        # Utility function to combine features into a single string with underscores
        feature_a = X[:, 0].astype(str)
        combined_features = feature_a
        for i in range(1, X.shape[1]):  # Start from 1 since 0 is already included
            feature = X[:, i].astype(str)
            combined_features = np.core.defchararray.add(combined_features, "_")
            combined_features = np.core.defchararray.add(combined_features, feature)
        return combined_features


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    y = data.target

    species = np.array(["setosa", "versicolor", "virginica"])[y]
    petal_length = X[:, 2].reshape(-1, 1)  # Petal length
    X_with_species = np.hstack([petal_length, species.reshape(-1, 1)])

    transformer = FeatureCrossBinaryEncoder(mode="ordinal")
    transformer.fit(X_with_species)
    transformed = transformer.transform(X_with_species)
    print(transformed[:5])
