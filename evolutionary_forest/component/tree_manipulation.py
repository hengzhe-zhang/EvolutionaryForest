import math
from functools import partial
from typing import List
import numpy as np  # noqa
from deap import gp
from deap.gp import PrimitiveTree, Terminal
from gplearn.functions import (
    _protected_division,
    _protected_log,
    _protected_sqrt,
    _sigmoid,
)
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    LabelEncoder,
    KBinsDiscretizer,
    OrdinalEncoder,
)

from evolutionary_forest.component.primitive_functions import (
    analytical_log,
    analytical_quotient,
    radian_sin,
    radian_cos,
)
from evolutionary_forest.component.stgp.categorical_processor import *
from evolutionary_forest.component.stgp.fast_binary_encoder import BinaryEncoder
from evolutionary_forest.component.stgp.feature_crossing import (
    FeatureCrossBinaryEncoder,
)
from evolutionary_forest.component.stgp.shared_type import (
    Parameter,
    LearnedParameter,
    CategoricalFeature,
    FeatureLayer,
)
from evolutionary_forest.component.stgp.smooth_scaler import NearestValueTransformer
from evolutionary_forest.multigene_gp import quick_fill


def copy_categorical_features(data, categorical_features: list[bool]):
    return data, categorical_features
    if np.sum(categorical_features) > 0:
        # add categorical features to the end of the data
        cat_features = np.column_stack(
            [
                data[:, idx]
                for idx in range(len(categorical_features))
                if categorical_features[idx]
            ]
        )
        data = np.concatenate([data, cat_features], axis=1)
        categorical_features = categorical_features + [False] * cat_features.shape[1]
    return data, categorical_features


def standardize(x, scaler):
    # standardize the input
    return scaler.transform(x.reshape(-1, 1)).flatten()


def min_max_scaler(x, scaler):
    # standardize the input
    return scaler.transform(x.reshape(-1, 1)).flatten()


def robust_scaler(x, scaler):
    return scaler.transform(x.reshape(-1, 1)).flatten()


def clip(x, min_val, max_val):
    # standardize the input
    x = np.clip(x, min_val, max_val)
    return x


def gaussian_quantile_transformer(x, qt: QuantileTransformer):
    return qt.transform(x.reshape(-1, 1)).flatten()


def normal_quantile_transformer(x, qt: QuantileTransformer):
    return qt.transform(x.reshape(-1, 1)).flatten()


def groupby_mean(x, y, transformer: GroupByMeanTransformer):
    return transformer.transform(np.column_stack((x, y)))


def groupby_median(x, y, transformer: GroupByMedianTransformer):
    return transformer.transform(np.column_stack((x, y)))


def groupby_min(x, y, transformer: GroupByMinTransformer):
    return transformer.transform(np.column_stack((x, y)))


def groupby_max(x, y, transformer: GroupByMaxTransformer):
    return transformer.transform(np.column_stack((x, y)))


def groupby_count(x, y, transformer: GroupByCountTransformer):
    return transformer.transform(np.column_stack((x, y)))


def binary_feature_cross(x, y, transformer: FeatureCrossBinaryEncoder):
    return transformer.transform(np.column_stack((x, y)))


def triple_feature_cross(x, y, z, transformer: FeatureCrossBinaryEncoder):
    return transformer.transform(np.column_stack((x, y, z)))


def onehot_encoding(x, transformer: OneHotEncoder):
    return transformer.transform(x.reshape(-1, 1))


def binary_encoding(x, transformer: BinaryEncoder):
    return transformer.transform(x)


def ordinal_encoding(x, transformer: LabelEncoder):
    return transformer.transform(x.reshape(-1, 1)).flatten()


def binning(x, transformer: KBinsDiscretizer):
    return transformer.transform(x.reshape(-1, 1)).flatten()


def linear_layer(x, weights):
    result = (x @ weights).flatten()
    assert x.shape == result.shape
    return result


def fitting(function, data):
    data = np.nan_to_num(data, posinf=0, neginf=0)
    if function == standardize:
        sc = StandardScaler().fit(data[0].reshape(-1, 1))
        return (sc,)
    elif function == min_max_scaler:
        scaler = MinMaxScaler().fit(data[0].reshape(-1, 1))
        return (scaler,)
    elif function == robust_scaler:
        scaler = RobustScaler().fit(data[0].reshape(-1, 1))
        return (scaler,)
    elif function == linear_layer:
        return (np.random.randn(len(data[0]), len(data[0])),)
    elif function == gaussian_quantile_transformer:
        qt = QuantileTransformer(output_distribution="uniform")
        qt.fit(data[0].reshape(-1, 1))
        return (qt,)
    elif function == groupby_mean:
        transformer = GroupByMeanTransformer()
        transformer.fit(np.column_stack((data[0], data[1])))
        return (transformer,)
    elif function == groupby_median:
        transformer = GroupByMedianTransformer()
        transformer.fit(np.column_stack((data[0], data[1])))
        return (transformer,)
    elif function == groupby_min:
        transformer = GroupByMinTransformer()
        transformer.fit(np.column_stack((data[0], data[1])))
        return (transformer,)
    elif function == groupby_max:
        transformer = GroupByMaxTransformer()
        transformer.fit(np.column_stack((data[0], data[1])))
        return (transformer,)
    elif function == groupby_count:
        transformer = GroupByCountTransformer()
        transformer.fit(np.column_stack((data[0], data[1])))
        return (transformer,)
    elif function == normal_quantile_transformer:
        qt = QuantileTransformer(output_distribution="normal")
        qt.fit(data[0].reshape(-1, 1))
        return (qt,)
    elif function == binary_encoding:
        transformer = BinaryEncoder()
        transformer.fit(data[0])
        return (transformer,)
    elif function == onehot_encoding:
        transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformer.fit(data[0].reshape(-1, 1))
        return (transformer,)
    elif function == ordinal_encoding:
        transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        transformer.fit(data[0].reshape(-1, 1))
        return (transformer,)
    elif function == binary_feature_cross:
        transformer = FeatureCrossBinaryEncoder(mode="ordinal")
        transformer.fit(np.column_stack((data[0], data[1])))
        return (transformer,)
    elif function == triple_feature_cross:
        transformer = FeatureCrossBinaryEncoder(mode="binary")
        transformer.fit(np.column_stack((data[0], data[1], data[2])))
        return (transformer,)
    elif function == binning:
        transformer = KBinsDiscretizer(
            math.ceil(math.sqrt(len(data[0]))), encode="ordinal"
        )
        transformer.fit(data[0].reshape(-1, 1))
        return (transformer,)
    return smooth_fitting(function, data)


def identity(x):
    return x


def identity_categorical(x):
    return x


def get_typed_pset(
    shape, primitive_type, categorical_features: list[bool]
) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("MAIN", [float for _ in range(shape)], float, "ARG")
    if primitive_type.endswith("-Smooth"):
        if primitive_type.endswith("-Smooth-Analytical"):
            add_smooth_math_operators(pset, analytical_operators=True)
        else:
            add_smooth_math_operators(pset)
        pset.addEphemeralConstant("Parameter", lambda: Parameter(), Parameter)
        return pset
    if primitive_type.endswith("-Basic"):
        add_math_operators(pset)
        return pset
    add_math_operators(pset)
    add_scaling_primitives(pset)
    # pset.addPrimitive(binning, [float, Parameter], float)
    if primitive_type.endswith("-Categorical"):
        feature_types = [
            CategoricalFeature if categorical_features[idx] else float
            for idx in range(shape)
        ]
        pset = gp.PrimitiveSetTyped("MAIN", feature_types, FeatureLayer, "ARG")
        has_numerical_features = len(categorical_features) != np.sum(
            categorical_features
        )
        # has_numerical_features = True
        has_categorical_features = np.sum(categorical_features) != 0
        if has_numerical_features:
            pset.addPrimitive(identity, [float], FeatureLayer)
            add_math_operators(pset)
        if has_numerical_features and has_categorical_features:
            # must have categorical features
            pset.addPrimitive(
                groupby_mean, [CategoricalFeature, float, Parameter], float
            )
            pset.addPrimitive(
                groupby_median, [CategoricalFeature, float, Parameter], float
            )
            pset.addPrimitive(
                groupby_min, [CategoricalFeature, float, Parameter], float
            )
            pset.addPrimitive(
                groupby_max, [CategoricalFeature, float, Parameter], float
            )
            pset.addPrimitive(
                groupby_count, [CategoricalFeature, float, Parameter], float
            )
        if has_categorical_features:
            pset.addPrimitive(
                binary_feature_cross,
                [CategoricalFeature, CategoricalFeature, Parameter],
                CategoricalFeature,
            )
            # pset.addPrimitive(
            #     binary_feature_cross,
            #     [CategoricalFeature, CategoricalFeature, Parameter],
            #     FeatureLayer,
            # )
            # pset.addPrimitive(
            #     triple_feature_cross,
            #     [CategoricalFeature, CategoricalFeature, CategoricalFeature, Parameter],
            #     FeatureLayer,
            # )
            pset.addPrimitive(
                onehot_encoding, [CategoricalFeature, Parameter], FeatureLayer
            )
            pset.addPrimitive(ordinal_encoding, [CategoricalFeature, Parameter], float)
    pset.addEphemeralConstant("Parameter", lambda: Parameter(), Parameter)
    # pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1), float)
    return pset


def add_scaling_primitives(pset):
    pset.addPrimitive(standardize, [float, Parameter], float)
    pset.addPrimitive(min_max_scaler, [float, Parameter], float)
    pset.addPrimitive(robust_scaler, [float, Parameter], float)
    pset.addPrimitive(gaussian_quantile_transformer, [float, Parameter], float)
    pset.addPrimitive(normal_quantile_transformer, [float, Parameter], float)


def add_math_operators(pset):
    pset.addPrimitive(np.add, [float, float], float)
    pset.addPrimitive(np.subtract, [float, float], float)
    pset.addPrimitive(np.multiply, [float, float], float)
    pset.addPrimitive(_protected_division, [float, float], float)
    pset.addPrimitive(_protected_sqrt, [float], float)
    pset.addPrimitive(_protected_log, [float], float)
    pset.addPrimitive(_sigmoid, [float], float)
    pset.addPrimitive(np.minimum, [float, float], float)
    pset.addPrimitive(np.maximum, [float, float], float)
    pset.addPrimitive(np.square, [float], float)


def smooth_fitting(function, data):
    if function.func in [smooth_operator_1, smooth_operator_2]:
        transformer = NearestValueTransformer()
        transformer.fit(function.keywords["operator"](*data))
        return (transformer,)


def smooth_operator_1(x, trans: NearestValueTransformer, operator):
    return trans.transform(operator(x))


def smooth_operator_2(x, y, trans: NearestValueTransformer, operator):
    return trans.transform(operator(x, y))


def partial_wrapper(function, operator):
    partial_func = partial(function, operator=operator)
    partial_func.__name__ = operator.__name__
    return partial_func


def add_smooth_math_operators(pset, analytical_operators=False):
    pset.addPrimitive(
        partial_wrapper(smooth_operator_2, operator=np.add),
        [float, float, Parameter],
        float,
    )
    pset.addPrimitive(
        partial_wrapper(smooth_operator_2, operator=np.subtract),
        [float, float, Parameter],
        float,
    )
    pset.addPrimitive(
        partial_wrapper(smooth_operator_2, operator=np.multiply),
        [float, float, Parameter],
        float,
    )
    if not analytical_operators:
        pset.addPrimitive(
            partial_wrapper(smooth_operator_2, operator=_protected_division),
            [float, float, Parameter],
            float,
        )
        pset.addPrimitive(
            partial_wrapper(smooth_operator_1, operator=_protected_log),
            [float, Parameter],
            float,
        )
    pset.addPrimitive(
        partial_wrapper(smooth_operator_1, operator=_protected_sqrt),
        [float, Parameter],
        float,
    )
    if analytical_operators:
        pset.addPrimitive(
            partial_wrapper(smooth_operator_2, operator=analytical_quotient),
            [float, float, Parameter],
            float,
        )
        pset.addPrimitive(
            partial_wrapper(smooth_operator_1, operator=analytical_log),
            [float, Parameter],
            float,
        )
        pset.addPrimitive(
            partial_wrapper(smooth_operator_1, operator=radian_sin),
            [float, Parameter],
            float,
        )
        pset.addPrimitive(
            partial_wrapper(smooth_operator_1, operator=radian_cos),
            [float, Parameter],
            float,
        )
        pset.addPrimitive(
            partial_wrapper(smooth_operator_1, operator=np.abs),
            [float, Parameter],
            float,
        )
        pset.addPrimitive(
            partial_wrapper(smooth_operator_1, operator=np.negative),
            [float, Parameter],
            float,
        )
    pset.addPrimitive(
        partial_wrapper(smooth_operator_1, operator=_sigmoid), [float, Parameter], float
    )
    pset.addPrimitive(
        partial_wrapper(smooth_operator_2, operator=np.minimum),
        [float, float, Parameter],
        float,
    )
    pset.addPrimitive(
        partial_wrapper(smooth_operator_2, operator=np.maximum),
        [float, float, Parameter],
        float,
    )
    pset.addPrimitive(
        partial_wrapper(smooth_operator_1, operator=np.square),
        [float, Parameter],
        float,
    )


def multi_tree_evaluation_typed(
    gp_trees: List[PrimitiveTree],
    pset,
    data: np.ndarray,
):
    results = []
    for tree in gp_trees:
        result = quick_evaluate(tree, pset, data)
        if len(result.shape) == 2:
            # 2D array
            results.extend(list(result.T))
        else:
            results.append(result)
    results = quick_fill(results, data)
    return results.T


def quick_evaluate(expr: PrimitiveTree, pset, data, prefix="ARG"):
    result = None
    stack = []
    for idx, node in enumerate(expr):
        stack.append((node, [], idx))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args, node_idx = stack.pop()
            if isinstance(prim, gp.Primitive):
                arg_values, arg_indices = zip(*args) if args else ([], [])

                if any(isinstance(arg, Parameter) for arg in arg_values):
                    # Find the first Parameter instance and its index
                    param_index = next(
                        i
                        for i, arg in enumerate(arg_values)
                        if isinstance(arg, Parameter)
                    )
                    # Apply fitting function up to the first Parameter instance
                    parameters = fitting(
                        pset.context[prim.name], arg_values[:param_index]
                    )

                    # Update arguments with fitted parameters
                    updated_args = list(arg_values[:param_index]) + list(parameters)

                    # Update new_expr with fitted parameters
                    for p_id, parameter in enumerate(parameters):
                        assert isinstance(
                            expr[arg_indices[param_index + p_id]].value, Parameter
                        )
                        expr[arg_indices[param_index + p_id]] = gp.Terminal(
                            parameter, False, LearnedParameter
                        )

                    result = pset.context[prim.name](*updated_args)
                else:
                    result = pset.context[prim.name](*arg_values)
            elif isinstance(prim, gp.Terminal):
                if prefix in prim.name:
                    result = data[:, int(prim.name.replace(prefix, ""))]
                else:
                    result = prim.value
            else:
                raise Exception("Invalid node in the expression tree.")
            if len(stack) == 0:
                break
            stack[-1][1].append((result, node_idx))

    # Replace the original expression tree with the updated one
    return result


def revert_back(ind):
    for tree in ind.gene:
        # tree:PrimitiveSetTyped
        for i in range(len(tree)):
            if tree[i].ret == LearnedParameter:
                tree[i] = Terminal(Parameter(), False, Parameter)
