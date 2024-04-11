from typing import List

import numpy as np
from deap import gp
from deap.gp import PrimitiveTree, Terminal
from sklearn.preprocessing import (
    QuantileTransformer,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

from evolutionary_forest.component.primitive_functions import (
    protected_division,
    analytical_log,
)
from evolutionary_forest.component.shared_type import Parameter, LearnedParameter
from evolutionary_forest.multigene_gp import quick_fill


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


def quantile_transformer(x, qt: QuantileTransformer):
    return qt.transform(x.reshape(-1, 1)).flatten()


def normal_quantile_transformer(x, qt: QuantileTransformer):
    return qt.transform(x.reshape(-1, 1)).flatten()


def linear_layer(x, weights):
    result = (x @ weights).flatten()
    assert x.shape == result.shape
    return result


def fitting(function, data):
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
    elif function == quantile_transformer:
        qt = QuantileTransformer()
        qt.fit(data[0].reshape(-1, 1))
        return (qt,)
    elif function == normal_quantile_transformer:
        qt = QuantileTransformer(output_distribution="normal")
        qt.fit(data[0].reshape(-1, 1))
        return (qt,)


def get_typed_pset(shape):
    pset = gp.PrimitiveSetTyped("MAIN", [float for _ in range(shape)], float, "ARG")
    pset.addPrimitive(np.add, [float, float], float)
    pset.addPrimitive(np.subtract, [float, float], float)
    pset.addPrimitive(np.multiply, [float, float], float)
    pset.addPrimitive(protected_division, [float, float], float)
    pset.addPrimitive(np.minimum, [float, float], float)
    pset.addPrimitive(np.maximum, [float, float], float)
    pset.addPrimitive(analytical_log, [float], float)
    # pset.addPrimitive(linear_layer, [float, Parameter], float)
    pset.addPrimitive(standardize, [float, Parameter], float)
    pset.addPrimitive(min_max_scaler, [float, Parameter], float)
    pset.addPrimitive(robust_scaler, [float, Parameter], float)
    pset.addPrimitive(quantile_transformer, [float, Parameter], float)
    pset.addEphemeralConstant("Parameter", lambda: Parameter(), Parameter)
    # pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1), float)
    return pset


def multi_tree_evaluation_typed(
    gp_trees: List[PrimitiveTree],
    pset,
    data: np.ndarray,
):
    results = []
    for tree in gp_trees:
        result = quick_evaluate(tree, pset, data)
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
