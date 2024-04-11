import copy
from typing import List

import numpy as np
from deap import gp
from deap.gp import PrimitiveTree

from evolutionary_forest.multigene_gp import quick_fill


def standardize(x, mean, std):
    # standardize the input
    x = (x - mean) / std
    return x


def fitting(function, data):
    if function == standardize:
        return np.mean(data), np.std(data)


class Parameter:
    pass


def get_typed_pset(shape):
    pset = gp.PrimitiveSetTyped("MAIN", [float for _ in range(shape)], float, "X")
    pset.addPrimitive(np.add, [float, float], float)
    pset.addPrimitive(np.subtract, [float, float], float)
    pset.addPrimitive(np.multiply, [float, float], float)
    pset.addPrimitive(standardize, [float, Parameter, Parameter], float)
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
    results = quick_fill(np.array(results), data)
    return results


def quick_evaluate(expr: PrimitiveTree, pset, data, prefix="X"):
    new_expr = copy.deepcopy(expr)

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
                        new_expr[node_idx + param_index + p_id] = gp.Terminal(
                            parameter, False, type(parameter)
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
    expr[:] = new_expr[:]
    return result
