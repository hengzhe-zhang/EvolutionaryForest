from sympy import preorder_traversal, parse_expr, latex

from sympy import Float


def count_nodes(expr):
    return sum(1 for _ in preorder_traversal(expr))


def model_simplification(sympy_model):
    # replace literal pi string with symbolic pi, then parse
    expr = parse_expr(str(sympy_model).replace("3.141592653589793", "pi"))
    # round every Float atom to 3 significant digits
    expr = expr.xreplace({f: Float(f, 3) for f in expr.atoms(Float)})
    # final rename
    return str(expr).replace("ARG", "X")


if __name__ == "__main__":
    print(latex(parse_expr(model_simplification("X1/Max(X0,X1)"))))
