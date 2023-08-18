import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def get_feature_type(x, include_binary=False):
    # x = x[~np.isnan(x)]
    if not check_if_all_integers(x):
        return 'continuous'
    else:
        unique_values = np.unique(x)
        if unique_values.size > 5:
            return 'continuous'
        if include_binary:
            if unique_values.size == 2:
                return 'binary'
        return 'categorical'


def check_if_all_integers(x):
    "check a NumPy array is made of all integers."
    unique_values = np.unique(x)
    return all(float(i).is_integer() for i in unique_values)


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, include_binary=False):
        self.include_binary = include_binary
        self.encodings = {}
        self.global_means = {}

    def fit(self, X, y):
        for column in range(X.shape[1]):
            feature = X[:, column]
            feature_type = get_feature_type(feature, self.include_binary)
            if feature_type == 'categorical' or (feature_type == 'binary' and self.include_binary):
                encoding = {}
                for unique_value in np.unique(feature):
                    encoding[unique_value] = y[feature == unique_value].mean()
                self.encodings[column] = encoding
                self.global_means[column] = y.mean()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column, encoding in self.encodings.items():
            global_mean = self.global_means[column]
            for i in range(X.shape[0]):
                value = X[i, column]
                X_transformed[i, column] = encoding.get(value, global_mean)
        return X_transformed
