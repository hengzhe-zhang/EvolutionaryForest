import numpy as np
from category_encoders import OrdinalEncoder, OneHotEncoder
from category_encoders import TargetEncoder, LeaveOneOutEncoder, BinaryEncoder, CatBoostEncoder, PolynomialEncoder, \
    BackwardDifferenceEncoder, CountEncoder, HashingEncoder, HelmertEncoder, JamesSteinEncoder, \
    MEstimateEncoder, QuantileEncoder, SumEncoder, SummaryEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
        # # Feature Transformation based on different types of features
        # types = type_detection(X)
        # num_cols = []
        # bool_cols = []
        # cat_cols = []
        # for i, t in enumerate(types):
        #     if t == NumericalFeature:
        #         num_cols.append(i)
        #     elif t == BooleanFeature:
        #         bool_cols.append(i)
        #     else:
        #         cat_cols.append(i)

        encoder = {
            'TargetEncoder': TargetEncoder(),
            'LeaveOneOutEncoder': LeaveOneOutEncoder(),
            'BinaryEncoder': BinaryEncoder(),
            'CatBoostEncoder': CatBoostEncoder(),
            'PolynomialEncoder': PolynomialEncoder(),
            'BackwardDifferenceEncoder': BackwardDifferenceEncoder(),
            'CountEncoder': CountEncoder(),
            'HashingEncoder': HashingEncoder(),
            'HelmertEncoder': HelmertEncoder(),
            'JamesSteinEncoder': JamesSteinEncoder(),
            'MEstimateEncoder': MEstimateEncoder(),
            'OneHotEncoder': OneHotEncoder(),
            'OrdinalEncoder': OrdinalEncoder(),
            'QuantileEncoder': QuantileEncoder(),
            'SumEncoder': SumEncoder(),
            'SummaryEncoder': SummaryEncoder(),
            # 'GLMMEncoder': GLMMEncoder(), # Unavailable
        }[self.encoding_scheme]
        num_transformer = Pipeline([
            ('Imputer', SimpleImputer(strategy='mean')),
            ('Encoder', StandardScaler()),
        ])
        cat_transformer = Pipeline(
            [('Imputer', SimpleImputer(strategy='constant', fill_value='missing')),
             ('Encoder', encoder),
             ('Scaler', StandardScaler())]
        )
        transformer = ColumnTransformer(
            transformers=[
                ('num', num_transformer, selector(dtype_include='float64')),
                ('cat', cat_transformer, selector(dtype_include='category')),
                ('obj', cat_transformer, selector(dtype_include='object')),
            ])

        self.x_scaler = transformer
        self.x_scaler.fit(X, y)
        return self

    def transform(self, X):
        return self.x_scaler.transform(X)


if __name__ == '__main__':
    pass
