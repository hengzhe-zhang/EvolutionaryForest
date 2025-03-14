import inspect

import numpy as np
from deap.gp import PrimitiveSet


def add_generated_function_to_pset(
    generated_functions, function_arity, pset: PrimitiveSet
):
    # Register dynamically generated functions
    for name, func in generated_functions.items():
        arity = function_arity[name]  # Get function's arity
        pset.addPrimitive(
            func, arity, name=name
        )  # Register with correct argument count


def add_function_to_pset(parsed_data):
    # Convert string functions into actual Python functions
    generated_functions = {}
    function_arity = {}  # Store arity of each function

    for name, func_str in parsed_data.items():
        # 生成实际的函数对象
        generated_func = eval(func_str, {"np": np})

        # 获取参数个数
        arity = len(inspect.signature(generated_func).parameters)

        # 存储函数及其参数个数
        generated_functions[name] = generated_func
        function_arity[name] = arity

    return generated_functions, function_arity
