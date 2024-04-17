import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from evolutionary_forest.component.stgp.fast_binary_encoder import BinaryEncoder


class FeatureCrossBinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = BinaryEncoder()

    def fit(self, X, y=None):
        feature_a = X[:, 0].astype(str)
        combined_features = feature_a
        for i in range(X.shape[1]):
            feature = X[:, i].astype(str)
            combined_features = np.core.defchararray.add(combined_features, "_")
            combined_features = np.core.defchararray.add(combined_features, feature)
        self.encoder.fit(combined_features)
        return self

    def transform(self, X):
        # Recreate the combined feature in the same way as during fit
        feature_a = X[:, 0].astype(str)
        combined_features = feature_a
        for i in range(X.shape[1]):
            feature = X[:, i].astype(str)
            combined_features = np.core.defchararray.add(combined_features, "_")
            combined_features = np.core.defchararray.add(combined_features, feature)
        return self.encoder.transform(combined_features)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data
    y = data.target

    species = np.array(["setosa", "versicolor", "virginica"])[y]
    petal_length = X[:, 2].reshape(-1, 1)  # Petal length
    X_with_species = np.hstack([petal_length, species.reshape(-1, 1)])

    transformer = FeatureCrossBinaryEncoder()
    transformer.fit(X_with_species)
    transformed = transformer.transform(X_with_species)
    print(transformed[:5])
