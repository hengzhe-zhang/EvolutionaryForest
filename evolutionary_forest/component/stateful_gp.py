import enum
from abc import ABCMeta

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from category_encoders.utils import BaseEncoder
from sklearn.base import TransformerMixin, BaseEstimator


class DimensionTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape((-1, 1))


class TargetEncoderNumpy(TargetEncoder):
    def fit(self, X, y=None, **kwargs):
        X = pd.DataFrame(X).astype(str).sum(axis=1).to_frame()
        return super().fit(X, y, **kwargs)

    def transform(self, X, y=None, override_return_df=False):
        X = pd.DataFrame(X).astype(str).sum(axis=1).to_frame()
        return super().transform(X, y, override_return_df)


def make_class(function, arity=None, parameters=None):
    if parameters is None:
        parameters = {}

    class FeatureTransformer(TransformerMixin):
        def __init__(self, *args):
            if isinstance(function, type):
                self.function = function(**parameters)
            else:
                self.function = function
            self.args = args

        def fit(self, X, y=None, **fit_params):
            output = []
            for arg in self.args:
                if isinstance(arg, (int, float, enum.Enum)):
                    output.append(self.decision_function(X, arg))
                else:
                    output.append(arg.fit_transform(X, y).flatten())
            if isinstance(self.function, (BaseEstimator, BaseEncoder)):
                if len(output) == 1:
                    output = np.array(output).reshape(-1, 1)
                else:
                    output = np.array(output).T
                self.function.fit(output, y)
            return self

        def decision_function(self, X, x):
            if isinstance(x, int) or isinstance(x, float):
                return np.full(X.shape[0], x)
            elif isinstance(x, enum.Enum):
                return X[:, x.value]
            elif hasattr(x, "predict"):
                # inner regressors
                return x.predict(X)
            else:
                return x.transform(X).flatten()

        def transform(self, X):
            if isinstance(self.function, (BaseEstimator, BaseEncoder)):
                output = [self.decision_function(X, x) for x in self.args]
                if len(output) == 1:
                    output = np.array(output).reshape(-1, 1)
                else:
                    output = np.array(output).T
                if hasattr(self.function, "predict"):
                    output = self.function.predict(output)
                else:
                    output = self.function.transform(output).flatten()
            else:
                output = self.function(
                    *[self.decision_function(X, x) for x in self.args]
                )
            return np.reshape(output, (-1, 1))

    if isinstance(function, ABCMeta):
        FeatureTransformer.__name__ = function.__name__
    else:
        FeatureTransformer.__name__ = function.__name__
    if arity != None:
        FeatureTransformer.__name__ = FeatureTransformer.__name__ + f"_{arity}"
    return FeatureTransformer


class DummyTransformer(TransformerMixin, BaseEstimator):
    """Dummy transformer to output the first column."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, 0]
