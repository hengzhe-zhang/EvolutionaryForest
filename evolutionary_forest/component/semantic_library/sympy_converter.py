import sympy
from deap import gp


def tree_to_sympy(tree, pset):
    """
    Converts a DEAP GP tree to a SymPy expression.

    This function is designed to work with the specific iteration pattern where
    nodes are processed from a stack after all their children have been processed.

    :param tree: The DEAP GP PrimitiveTree to convert.
    :param pset: The PrimitiveSet used to create the tree. This is needed to
                 correctly identify variables (arguments) vs. constants.
    :return: A SymPy expression equivalent to the input tree.
    """

    # --- 1. Mapping from DEAP primitive names to SymPy functions ---
    # This dictionary is crucial for the conversion. It translates the name
    # of a primitive in your pset to the corresponding SymPy operation.
    SYMPY_MAP = {
        # Standard operators
        "add": sympy.Add,
        "subtract": lambda x, y: x - y,
        "multiply": sympy.Mul,
        "neg": lambda x: -x,
        "tanh": sympy.tanh,
        "square": lambda x: x**2,
        "minimum": sympy.Min,
        "maximum": sympy.Max,
        # Custom protected operators
        "protected_div": lambda x, y: x / y,
        "protected_log": lambda x: sympy.log(sympy.Abs(x) + 1),
        "protected_sqrt": lambda x: sympy.sqrt(sympy.Abs(x)),
        # Custom trigonometric functions
        "sin_pi": lambda x: sympy.sin(sympy.pi * x),
        "cos_pi": lambda x: sympy.cos(sympy.pi * x),
    }

    # --- 2. The Conversion Logic (adapted from your snippet) ---
    stack = []

    # This will hold the final, complete SymPy expression.
    result_expr = None

    # We iterate through the tree nodes in DEAP's default prefix order.
    for node in tree:
        # Push each node onto the stack, paired with a list for its children's
        # processed expressions.
        stack.append((node, []))

        # This inner loop runs whenever a node on the stack has collected all
        # of its required arguments (children).
        while len(stack[-1][1]) == stack[-1][0].arity:
            # Pop the node and its processed children (as SymPy expressions).
            prim, args = stack.pop()

            # --- Convert the current node to a SymPy expression ---

            if isinstance(prim, gp.Primitive):
                # The node is a function.
                if prim.name not in SYMPY_MAP:
                    raise ValueError(
                        f"Primitive '{prim.name}' is not in the SYMPY_MAP."
                    )

                sympy_func = SYMPY_MAP[prim.name]
                result_expr = sympy_func(*args)

            elif isinstance(prim, gp.Terminal):
                # The node is a variable or a constant value.
                # The 'value' attribute holds its name (e.g., 'ARG0') or value (e.g., 3.14).

                # Check if the terminal is a defined input variable.
                if prim.value in pset.arguments:
                    result_expr = sympy.Symbol(str(prim.value))
                else:
                    # It's a constant (e.g., from addTerminal or EphemeralConstant).
                    # sympify() safely converts it to a SymPy numeric type.
                    result_expr = sympy.sympify(prim.value)

            else:
                # Should not be reached with a valid DEAP tree.
                raise TypeError(f"Unknown node type encountered: {type(prim)}")

            if not stack:
                # If the stack is empty, we have just processed the root node.
                # The conversion is complete.
                break

                # Otherwise, this expression is a child of the node now on top of the stack.
            # Append it to the parent's argument list.
            stack[-1][1].append(result_expr)

    if result_expr is None:
        raise ValueError("Conversion failed. The tree may be empty or invalid.")

    return result_expr
