import json
import re

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


def add_function_to_pset(json_str):
    # Parse JSON
    parsed_data = json.loads(json_str)

    # Convert string functions into actual Python functions
    generated_functions = {}
    function_arity = {}  # Store arity of each function

    for name, func_str in parsed_data["generated_functions"].items():
        # Extract parameter count from lambda expression
        match = re.match(
            r"lambda\s*\((.*?)\):", func_str
        )  # Extracts "x, y" from "lambda x, y: ..."
        if match:
            param_list = match.group(1).replace(" ", "").split(",")  # Split parameters
            arity = len(param_list)  # Number of parameters
        else:
            raise ValueError(f"Could not determine arity for function: {name}")

        # Store arity
        function_arity[name] = arity

        # Convert string to function
        generated_functions[name] = eval(func_str, {"np": np})  # Safe sandboxing

    return function_arity, generated_functions
