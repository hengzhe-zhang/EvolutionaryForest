import numpy as np
from category_encoders import OrdinalEncoder, OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class GeneralFeature:
    pass


class CategoricalFeature:
    pass


class BooleanFeature:
    pass


class NumericalFeature:
    pass


class RandomSeed:
    pass


def check_if_all_integers(x):
    "check a pandas.Series is made of all integers."
    return all(float(i).is_integer() for i in np.unique(x))


def type_detection(X):
    input_arr = []
    for x in range(X.shape[1]):
        v = X[:, x]
        if len(np.unique(v)) == 2:
            input_arr.append(BooleanFeature)
        elif len(np.unique(v)) <= 10 and check_if_all_integers(v):
            input_arr.append(CategoricalFeature)
        else:
            input_arr.append(NumericalFeature)
    return input_arr


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_scheme=None):
        self.encoding_scheme = encoding_scheme

    def fit(self, X, y=None, **fit_params):
        from category_encoders import (
            TargetEncoder,
            LeaveOneOutEncoder,
            BinaryEncoder,
            CatBoostEncoder,
            PolynomialEncoder,
            BackwardDifferenceEncoder,
            CountEncoder,
            HashingEncoder,
            HelmertEncoder,
            JamesSteinEncoder,
            MEstimateEncoder,
            QuantileEncoder,
            SumEncoder,
            SummaryEncoder,
        )

        encoder = {
            "TargetEncoder": TargetEncoder(),
            "LeaveOneOutEncoder": LeaveOneOutEncoder(),
            "BinaryEncoder": BinaryEncoder(),
            "CatBoostEncoder": CatBoostEncoder(),
            "PolynomialEncoder": PolynomialEncoder(),
            "BackwardDifferenceEncoder": BackwardDifferenceEncoder(),
            "CountEncoder": CountEncoder(),
            "HashingEncoder": HashingEncoder(),
            "HelmertEncoder": HelmertEncoder(),
            "JamesSteinEncoder": JamesSteinEncoder(),
            "MEstimateEncoder": MEstimateEncoder(),
            "OneHotEncoder": OneHotEncoder(),
            "OrdinalEncoder": OrdinalEncoder(),
            "QuantileEncoder": QuantileEncoder(),
            "SumEncoder": SumEncoder(),
            "SummaryEncoder": SummaryEncoder(),
        }[self.encoding_scheme]
        num_transformer = Pipeline(
            [
                ("Imputer", SimpleImputer(strategy="mean")),
                ("Encoder", StandardScaler()),
            ]
        )
        cat_transformer = Pipeline(
            [
                ("Imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("Encoder", encoder),
                ("Scaler", StandardScaler()),
            ]
        )
        transformer = ColumnTransformer(
            transformers=[
                ("num", num_transformer, selector(dtype_include="float64")),
                ("cat", cat_transformer, selector(dtype_include="category")),
                ("obj", cat_transformer, selector(dtype_include="object")),
            ]
        )

        self.x_scaler = transformer
        self.x_scaler.fit(X, y)
        return self

    def transform(self, X):
        return self.x_scaler.transform(X)


class DummyScaler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X):
        return X


class StandardScalerWithMinMaxScaler(TransformerMixin):
    """
    Three kinds:
    MixMax
    Standard
    Non-Scaling
    """

    def __init__(self):
        self.std_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.unique_indices = None  # To store the unique column indices

    def fit(self, X, y=None):
        self.std_scaler.fit(X)
        self.minmax_scaler.fit(X)

        # During fitting, concatenate the data and find unique columns
        combined_data = np.hstack(
            (X, self.std_scaler.transform(X), self.minmax_scaler.transform(X))
        )

        # Find unique columns and store the indices for use in transform
        _, self.unique_indices = np.unique(combined_data, axis=1, return_index=True)
        self.unique_indices = np.sort(
            self.unique_indices
        )  # Ensure indices are sorted for consistency

        return self

    def transform(self, X, y=None):
        # Apply transformations using pre-fitted scalers
        combined_data = np.hstack(
            (X, self.std_scaler.transform(X), self.minmax_scaler.transform(X))
        )

        # Select only the unique columns identified during the fitting stage
        unique_data = combined_data[:, self.unique_indices]

        return unique_data


class StandardScalerWithMinMaxScalerAndBounds(StandardScalerWithMinMaxScaler):
    """
    Extends StandardScalerWithMinMaxScaler to include feature bounding.
    """

    def __init__(self):
        super().__init__()
        self.feature_min_ = None
        self.feature_max_ = None

    def fit(self, X, y=None):
        self.feature_min_ = np.min(X, axis=0)
        self.feature_max_ = np.max(X, axis=0)
        super().fit(X, y)
        return self

    def transform(self, X, y=None):
        X = np.clip(X, self.feature_min_, self.feature_max_)
        return super().transform(X)


class RealMLPPreprocessor(BaseEstimator, TransformerMixin):
    """
    RealMLP-style preprocessing for numeric features only.
    Implements quantile-based scaling and smooth clipping from Holzmüller et al. (2025).
    """

    def __init__(self):
        self.medians_ = None
        self.scales_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        self.medians_ = np.zeros(n_features)
        self.scales_ = np.zeros(n_features)

        for j in range(n_features):
            col = X[:, j]
            q0, q1_4, q1_2, q3_4, q1 = np.percentile(col, [0, 25, 50, 75, 100])

            if q3_4 != q1_4:
                s = 1.0 / (q3_4 - q1_4)  # RobustScaler-like
            elif q1 != q0:
                s = 2.0 / (q1 - q0)  # MinMaxScaler-like
            else:
                s = 0.0  # constant column

            self.medians_[j] = q1_2
            self.scales_[j] = s
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.medians_) * self.scales_
        # smooth clipping to roughly (-3, 3)
        X_clipped = X_scaled / np.sqrt(1 + (X_scaled / 3) ** 2)
        return X_clipped


if __name__ == "__main__":
    pass
