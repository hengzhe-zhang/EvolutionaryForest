import math
from functools import partial
from typing import List, Callable

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

from evolutionary_forest.component.configuration import EvaluationConfiguration
from evolutionary_forest.component.primitive_functions import (
    analytical_log,
    analytical_quotient,
    sin_pi,
    cos_pi,
    sqrt_signed,
    analytical_log_signed,
    protected_division,
    protect_sqrt,
    protect_log,
    abs_log,
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
from evolutionary_forest.component.post_processing.value_alignment import quick_fill


def check_is_scaler(x):
    return isinstance(x, float) or (isinstance(x, np.ndarray) and x.size == 1)


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


def is_inhomogeneous(arr):
    # Attempt to get the size of each element
    sizes = np.array([len(item) if isinstance(item, np.ndarray) else 1 for item in arr])
    # Check if all sizes are the same
    return len(np.unique(sizes))


def fitting(function, data):
    if is_inhomogeneous(data):
        data = list(data)
        for c in range(len(data)):
            data[c] = np.nan_to_num(data[c], posinf=0, neginf=0)
    else:
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
    if primitive_type.endswith("Smooth"):
        # Smooth operators, just have parameters, but does not have types
        if len(primitive_type.split("-")) == 3:
            flag = primitive_type.split("-")[1]
        else:
            flag = ""
        add_smooth_math_operators(pset, flag)
        pset.addEphemeralConstant("Parameter", lambda: Parameter(), Parameter)
        return pset
    flag = primitive_type.split("-")[1]
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
        add_scaling_primitives(pset)
        if has_numerical_features:
            pset.addPrimitive(identity, [float], FeatureLayer)
            add_math_operators(pset, flag)
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
            pset.addPrimitive(
                onehot_encoding, [CategoricalFeature, Parameter], FeatureLayer
            )
            pset.addPrimitive(
                binary_encoding, [CategoricalFeature, Parameter], FeatureLayer
            )
            pset.addPrimitive(
                ordinal_encoding, [CategoricalFeature, Parameter], FeatureLayer
            )
    else:
        add_math_operators(pset, flag)
        add_scaling_primitives(pset)
    pset.addEphemeralConstant("Parameter", lambda: Parameter(), Parameter)
    # pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1), float)
    return pset


def add_scaling_primitives(pset):
    pset.addPrimitive(standardize, [float, Parameter], float)
    pset.addPrimitive(min_max_scaler, [float, Parameter], float)
    pset.addPrimitive(robust_scaler, [float, Parameter], float)
    pset.addPrimitive(gaussian_quantile_transformer, [float, Parameter], float)
    pset.addPrimitive(normal_quantile_transformer, [float, Parameter], float)


def add_math_operators(pset, flag):
    operators = flag.split(",")
    tools = {
        "Add": (
            np.add,
            [float, float],
            float,
        ),
        "Sub": (
            np.subtract,
            [float, float],
            float,
        ),
        "Mul": (
            np.multiply,
            [float, float],
            float,
        ),
        "Div": (
            protected_division,
            [float, float],
            float,
        ),
        "Sqrt": (
            protect_sqrt,
            [float],
            float,
        ),
        "SinPi": (
            sin_pi,
            [float],
            float,
        ),
        "CosPi": (
            cos_pi,
            [float],
            float,
        ),
        "Abs": (
            np.abs,
            [float],
            float,
        ),
        "Square": (
            np.square,
            [float],
            float,
        ),
        "Min": (
            np.minimum,
            [float, float],
            float,
        ),
        "Max": (
            np.maximum,
            [float, float],
            float,
        ),
        "Neg": (
            np.negative,
            [float],
            float,
        ),
        "AQ": (
            analytical_quotient,
            [float, float],
            float,
        ),
        "AbsLog": (
            abs_log,
            [float],
            float,
        ),
    }
    for operator in operators:
        a, b, c = tools[operator]
        # a.__name__ = operator
        pset.addPrimitive(a, b, c)


def smooth_fitting(function, data):
    if function.func in [smooth_operator_1, smooth_operator_2]:
        transformer = NearestValueTransformer()
        transformer.fit(function.keywords["operator"](*data))
        return (transformer,)


def smooth_operator_1(x, trans: NearestValueTransformer, operator):
    return trans.transform(operator(x))
    # return operator(x)


def smooth_operator_2(x, y, trans: NearestValueTransformer, operator):
    return trans.transform(operator(x, y))
    # return operator(x, y)


def partial_wrapper(function, operator):
    partial_func = partial(function, operator=operator)
    partial_func.__name__ = operator.__name__
    return partial_func


def add_smooth_math_operators(pset, flag):
    operators = flag.split(",")
    tools = {
        "Add": (
            partial_wrapper(smooth_operator_2, operator=np.add),
            [float, float, Parameter],
            float,
        ),
        "Sub": (
            partial_wrapper(smooth_operator_2, operator=np.subtract),
            [float, float, Parameter],
            float,
        ),
        "Mul": (
            partial_wrapper(smooth_operator_2, operator=np.multiply),
            [float, float, Parameter],
            float,
        ),
        "Div": (
            partial_wrapper(smooth_operator_2, operator=_protected_division),
            [float, float, Parameter],
            float,
        ),
        "Sqrt": (
            partial_wrapper(smooth_operator_1, operator=_protected_sqrt),
            [float, Parameter],
            float,
        ),
        "Sqrt+": (
            partial_wrapper(smooth_operator_1, operator=sqrt_signed),
            [float, Parameter],
            float,
        ),
        "Log": (
            partial_wrapper(smooth_operator_1, operator=_protected_log),
            [float, Parameter],
            float,
        ),
        "RSin": (
            partial_wrapper(smooth_operator_1, operator=sin_pi),
            [float, Parameter],
            float,
        ),
        "RCos": (
            partial_wrapper(smooth_operator_1, operator=cos_pi),
            [float, Parameter],
            float,
        ),
        "Abs": (
            partial_wrapper(smooth_operator_1, operator=np.abs),
            [float, Parameter],
            float,
        ),
        "Sigmoid": (
            partial_wrapper(smooth_operator_1, operator=_sigmoid),
            [float, Parameter],
            float,
        ),
        "Square": (
            partial_wrapper(smooth_operator_1, operator=np.square),
            [float, Parameter],
            float,
        ),
        "Min": (
            partial_wrapper(smooth_operator_2, operator=np.minimum),
            [float, float, Parameter],
            float,
        ),
        "Max": (
            partial_wrapper(smooth_operator_2, operator=np.maximum),
            [float, float, Parameter],
            float,
        ),
        "Neg": (
            partial_wrapper(smooth_operator_1, operator=np.negative),
            [float, Parameter],
            float,
        ),
        "AQ": (
            partial_wrapper(smooth_operator_2, operator=analytical_quotient),
            [float, float, Parameter],
            float,
        ),
        "ALog": (
            partial_wrapper(smooth_operator_1, operator=analytical_log),
            [float, Parameter],
            float,
        ),
        "ALog+": (
            partial_wrapper(smooth_operator_1, operator=analytical_log_signed),
            [float, Parameter],
            float,
        ),
    }
    for operator in operators:
        a, b, c = tools[operator]
        a.__name__ = operator
        pset.addPrimitive(a, b, c)


def multi_tree_evaluation_typed(
    gp_trees: List[PrimitiveTree],
    pset,
    data: np.ndarray,
    evaluation_configuration: EvaluationConfiguration,
):
    results = []
    feature_numbers = []
    for tree in gp_trees:
        result = quick_evaluate(
            tree, pset, data, evaluation_configuration=evaluation_configuration
        )
        if isinstance(result, np.ndarray) and len(result.shape) == 2:
            # 2D array
            features = list(result.T)
            results.extend(features)
            feature_numbers.append(len(features))
        else:
            results.append(result)
            feature_numbers.append(1)
    results = quick_fill(results, data)
    return results.T, feature_numbers


def quick_evaluate(
    expr: PrimitiveTree,
    pset,
    data,
    prefix="ARG",
    evaluation_configuration: EvaluationConfiguration = None,
):
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
                if (
                    evaluation_configuration is not None
                    and evaluation_configuration.semantic_library is not None
                ):
                    # Add subtree to semantic lib
                    if isinstance(result, np.ndarray) and result.size > 1:
                        evaluation_configuration.semantic_library.append_subtree(
                            result, PrimitiveTree(expr[expr.searchSubtree(node_idx)])
                        )
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
