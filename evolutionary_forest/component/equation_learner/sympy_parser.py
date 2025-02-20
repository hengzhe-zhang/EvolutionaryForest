import ast
from deap import gp

DEBUG = True

# Mapping from Python AST operator types to primitive names.
BINARY_OPS = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
}

UNARY_OPS = {
    ast.USub: "neg",
    ast.UAdd: "pos",
}


class ConstTerminal:
    """
    A wrapper for numeric constants so that they have an 'arity' attribute and a format() method,
    as required by DEAP GP.
    """

    def __init__(self, value):
        self.value = value
        self.arity = 0

    def format(self, *args):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"ConstTerminal({self.value})"


def ast_to_gp(node, primitive_dict, var_map, pset):
    """
    Recursively converts a Python AST node into a list of DEAP GP tokens (primitives and terminals)
    in prefix order. For variable names (e.g. "x1") it retrieves the corresponding terminal from pset.terminals[object].
    """
    if DEBUG:
        print("Processing node:", ast.dump(node))

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in BINARY_OPS:
            raise ValueError("Unsupported binary operator: {}".format(op_type))
        op_name = BINARY_OPS[op_type]
        if op_name not in primitive_dict:
            raise ValueError("Operator '{}' not found in the primitive set.".format(op_name))
        prim = primitive_dict[op_name]
        if DEBUG:
            print(f"BinaryOp: Using primitive '{prim.name}' for operator '{op_name}'")
        left_tokens = ast_to_gp(node.left, primitive_dict, var_map, pset)
        right_tokens = ast_to_gp(node.right, primitive_dict, var_map, pset)
        return [prim] + left_tokens + right_tokens

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPS:
            raise ValueError("Unsupported unary operator: {}".format(op_type))
        op_name = UNARY_OPS[op_type]
        if op_name not in primitive_dict:
            raise ValueError("Operator '{}' not found in the primitive set.".format(op_name))
        prim = primitive_dict[op_name]
        if DEBUG:
            print(f"UnaryOp: Using primitive '{prim.name}' for operator '{op_name}'")
        operand_tokens = ast_to_gp(node.operand, primitive_dict, var_map, pset)
        return [prim] + operand_tokens

    elif isinstance(node, ast.Call):
        # Handle function calls such as sin(x1) or exp(x2).
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are supported.")
        func_name = node.func.id
        if func_name not in primitive_dict:
            raise ValueError("Function '{}' is not in the primitive set.".format(func_name))
        prim = primitive_dict[func_name]
        if DEBUG:
            print(f"Call: Using primitive '{prim.name}' for function call '{func_name}'")
        tokens = [prim]
        for arg in node.args:
            tokens.extend(ast_to_gp(arg, primitive_dict, var_map, pset))
        return tokens

    elif isinstance(node, ast.Name):
        # For a variable node (e.g. "x1"), use var_map to get its corresponding pset argument name,
        # then search pset.terminals[object] for a terminal with that name.
        var_id = node.id
        if var_id in var_map:
            arg_name = var_map[var_id]
            terminal = None
            for term in pset.terminals[object]:
                if hasattr(term, "name") and term.name == arg_name:
                    terminal = term
                    break
            if terminal is None:
                raise ValueError("Could not find terminal for variable {}".format(arg_name))
            if DEBUG:
                print(f"Name: Mapped variable '{var_id}' to terminal {terminal}")
            return [terminal]
        else:
            raise ValueError(f"Variable {var_id} not found in var_map")

    elif isinstance(node, ast.Constant):  # For Python 3.8+
        if isinstance(node.value, (int, float)):
            token = ConstTerminal(node.value)
            if DEBUG:
                print(f"Constant: {token}")
            return [token]
        else:
            raise ValueError("Unsupported constant type: {}".format(type(node.value)))

    elif hasattr(ast, "Num") and isinstance(node, ast.Num):  # For Python < 3.8
        token = ConstTerminal(node.n)
        if DEBUG:
            print(f"Num: {token}")
        return [token]

    else:
        raise ValueError("Unsupported AST node type: {}".format(type(node)))


def convert_to_deap_gp(expr_str, pset):
    """
    Converts a symbolic expression string into a DEAP GP PrimitiveTree.

    The function parses the expression into an AST, converts it into a prefix list of tokens
    (using the primitives from pset.primitives[pset.ret] and terminals from pset.terminals[object]),
    and returns a gp.PrimitiveTree.
    """
    if DEBUG:
        print("Converting expression:", expr_str)
    tree = ast.parse(expr_str, mode="eval")
    if DEBUG:
        print("AST of expression:", ast.dump(tree))

    # Build a dictionary mapping primitive names (as defined in the pset) to their objects.
    primitive_dict = {}
    for prim in pset.primitives[pset.ret]:
        primitive_dict[prim.name] = prim
        if DEBUG:
            print(f"Primitive added: {prim.name} with arity {prim.arity}")

    # Build a variable mapping: map "x1" -> first argument, "x2" -> second argument, etc.
    var_map = {f"x{i + 1}": pset.arguments[i] for i in range(len(pset.arguments))}
    if DEBUG:
        print("Variable mapping:", var_map)

    tokens = ast_to_gp(tree.body, primitive_dict, var_map, pset)
    if DEBUG:
        print("Prefix tokens:", tokens)

    gp_tree = gp.PrimitiveTree(tokens)
    if DEBUG:
        print("Constructed DEAP GP tree:", gp_tree)

    return gp_tree


# Example usage:
if __name__ == "__main__":
    import operator, math
    from deap import gp
    import random

    # Define a primitive set with two arguments.
    pset = gp.PrimitiveSet("MAIN", 2)
    # Add primitives with names that match those used in the expression.
    pset.addPrimitive(operator.add, 2, name="add")
    pset.addPrimitive(operator.sub, 2, name="sub")
    pset.addPrimitive(operator.mul, 2, name="mul")
    pset.addPrimitive(operator.truediv, 2, name="div")
    pset.addPrimitive(pow, 2, name="pow")
    pset.addPrimitive(math.sin, 1, name="sin")
    pset.addPrimitive(math.cos, 1, name="cos")
    pset.addPrimitive(math.exp, 1, name="exp")
    pset.addPrimitive(math.log, 1, name="log")
    pset.addPrimitive(operator.neg, 1, name="neg")
    # pset.arguments will typically be something like ("ARG0", "ARG1")

    # Define a sample expression string.
    expr_str = "sin(x1) + x2**2"  # corresponds to sin(x1) + (x2)^2

    # Convert the expression to a DEAP GP tree.
    gp_tree = convert_to_deap_gp(expr_str, pset)
    print("DEAP GP tree:", gp_tree)
