from functools import partial

from evolutionary_forest.component.primitive_functions import *

simple_reduce = lambda function, *x: reduce(function, x)


def get_functions(p):
    primitive = {
        # Arithmetic operations
        "Add": (np.add, 2),  # Addition
        "Sub": (np.subtract, 2),  # Subtraction
        "Mul": (np.multiply, 2),  # Multiplication
        "Div": (
            protected_division,
            2,
        ),  # Protected Division for handling divide-by-zero errors
        "Add^3": (partial(simple_reduce, np.add), 3),  # Addition
        "Add^4": (partial(simple_reduce, np.add), 4),  # Addition
        "Add^5": (partial(simple_reduce, np.add), 5),  # Addition
        "Sub^3": (partial(simple_reduce, np.subtract), 3),  # Subtraction
        "Sub^4": (partial(simple_reduce, np.subtract), 4),  # Subtraction
        "Sub^5": (partial(simple_reduce, np.subtract), 5),  # Subtraction
        "Mul^3": (partial(simple_reduce, np.multiply), 3),  # Multiplication
        "Mul^4": (partial(simple_reduce, np.multiply), 4),  # Multiplication
        "Mul^5": (partial(simple_reduce, np.multiply), 5),  # Multiplication
        "Max^3": (partial(simple_reduce, np.maximum), 3),  # Maximum
        "Max^4": (partial(simple_reduce, np.maximum), 4),  # Maximum
        "Max^5": (partial(simple_reduce, np.maximum), 5),  # Maximum
        "Min^3": (partial(simple_reduce, np.minimum), 3),  # Minimum
        "Min^4": (partial(simple_reduce, np.minimum), 4),  # Minimum
        "Min^5": (partial(simple_reduce, np.minimum), 5),  # Minimum
        "Div^3": (
            protected_division,
            3,
        ),  # Protected Division for handling divide-by-zero errors
        "Div^4": (
            protected_division,
            4,
        ),  # Protected Division for handling divide-by-zero errors
        "Div^5": (
            protected_division,
            5,
        ),  # Protected Division for handling divide-by-zero errors
        # Mathematical functions
        "AQ": (
            analytical_quotient,
            2,
        ),  # Analytical Quotient for symbolic differentiation
        "AQ-Singed": (analytical_quotient_signed, 2),  # AQ
        "AQ^3": (partial(simple_reduce, analytical_quotient), 3),  # AQ
        "AQ^4": (partial(simple_reduce, analytical_quotient), 4),  # AQ
        "AQ^5": (partial(simple_reduce, analytical_quotient), 5),  # AQ
        "Sqrt": (protect_sqrt, 1),  # Protected square root for handling negative values
        "Sqrt-Signed": (
            sqrt_signed,
            1,
        ),  # Protected square root for handling negative values
        "ALog": (
            analytical_log,
            1,
        ),  # Analytical Logarithm for symbolic differentiation
        "ALog-Signed": (
            analytical_log_singed,
            1,
        ),
        "ALog10": (
            analytical_log10,
            1,
        ),  # Analytical Logarithm base 10 for symbolic differentiation
        "Sin": (np.sin, 1),  # Sine function
        "Cos": (np.cos, 1),  # Cosine function
        "RSin": (radian_sin, 1),  # Sine function
        "RCos": (radian_cos, 1),  # Cosine function
        "DSin": (degree_sin, 1),  # Sine function
        "DCos": (degree_cos, 1),  # Cosine function
        "Arcsin": (np.arcsin, 1),  # Cosine function
        "Arccos": (np.arccos, 1),  # Cosine function
        "Arctan": (np.arctan, 1),  # Arctangent function
        "Tanh": (np.tanh, 1),  # Hyperbolic tangent function
        "Cbrt": (np.cbrt, 1),  # Cube root function
        "Square": (np.square, 1),  # Square function
        "Square-Signed": (square_signed, 1),  # Square function
        "Cube": (cube, 1),  # Cube function
        "FourthPower": (fourth_power, 1),  # Square function
        "FifthPower": (fifth_power, 1),  # Cube function
        "LogXY": (protected_log_xy, 2),
        "Log": (protected_log, 1),
        "Log2": (protected_log2, 1),
        "Log10": (protected_log10, 1),
        "Exp2": (protected_exp2, 1),
        "Exp": (protected_exp, 1),
        "Inv": (protected_inverse, 1),
        "Equal": (equal, 2),
        "NotEqual": (not_equal, 2),
        # Comparison operations
        "GE": (greater_or_equal_than, 2),  # Greater than or equal to comparison
        "LE": (less_or_equal_than, 2),  # Less than or equal to comparison
        "GE3A": (greater_or_equal_than_triple_a, 3),
        "GE3B": (greater_or_equal_than_triple_b, 3),
        "GE3C": (greater_or_equal_than_triple_c, 3),
        "GE4A": (greater_or_equal_than_quadruple_a, 4),
        # Greater than or equal to comparison for quadruple
        "GE4B": (greater_or_equal_than_quadruple_b, 4),
        # Greater than or equal to comparison for quadruple
        "GE4C": (greater_or_equal_than_quadruple_c, 4),
        # Greater than or equal to comparison for quadruple
        "LT4A": (less_than_quadruple_a, 4),  # Less than comparison for quadruple
        "LT4B": (less_than_quadruple_b, 4),  # Less than comparison for quadruple
        "LT4C": (less_than_quadruple_c, 4),  # Less than comparison for quadruple
        "GE2A": (
            greater_or_equal_than_double_a,
            2,
        ),  # Greater than or equal to comparison for double
        "GE2B": (
            greater_or_equal_than_double_b,
            2,
        ),  # Greater than or equal to comparison for double
        "GE2C": (
            greater_or_equal_than_double_c,
            2,
        ),  # Greater than or equal to comparison for double
        "LT2A": (less_than_double_a, 2),  # Less than comparison for double
        "LT2B": (less_than_double_b, 2),  # Less than comparison for double
        "LT2C": (less_than_double_c, 2),  # Less than comparison for double
        # Other functions
        "Abs": (np.absolute, 1),  # Absolute value function
        "Max": (np.maximum, 2),  # Maximum function
        "Min": (np.minimum, 2),  # Minimum function
        "Mean": (np_mean, 2),  # Mean function
        "ArgMax": (partial(np.argmax, axis=0), 2),
        "ArgMin": (partial(np.argmin, axis=0), 2),
        "Neg": (
            np.negative,
            1,
        ),  # Unary negation function (i.e. negate the input value)
        "Sigmoid": (sigmoid, 1),  # Sigmoid activation function
        "Round": (np.round, 1),  # Round to the nearest integer
        "Floor": (np.floor, 1),  # Round to the nearest integer
        "Ceil": (np.ceil, 1),  # Round to the nearest integer
        "Residual": (residual, 1),  # Residual function for handling negative values
        "Relu": (relu, 1),  # Residual function for handling negative values
        "Gaussian": (gaussian, 1),
        "GaussianKernel": (gaussian_kernel, 2),
        "ExponentialKernel": (exponential_kernel, 2),
        "LeakyRelu": (leaky_relu, 1),  # Leaky ReLU activation function
        # GroupBY Features
        "GroupByMean": (groupby_mean, 2),
        "GroupByMedian": (groupby_median, 2),
        "GroupByMax": (groupby_max, 2),
        "GroupByMin": (groupby_min, 2),
        "GroupByCount": (groupby_count, 1),
        "GroupByVar": (groupby_variance, 2),
    }[p]
    return primitive


def get_differentiable_functions(p):
    primitive = {
        # Arithmetic operations
        "Add": (torch.add, 2),  # Addition
        "Sub": (torch.subtract, 2),  # Subtraction
        "Mul": (torch.multiply, 2),  # Multiplication
        "Div": (
            protected_division_torch,
            2,
        ),  # Protected Division for handling divide-by-zero errors
        "AQ": (analytical_quotient_torch, 2),
        "ALog": (analytical_quotient_torch, 2),
        # Analytical Quotient for symbolic differentiation
        "Sqrt": (protect_sqrt_torch, 1),
        # Protected square root for handling negative values
        "Sin": (torch.sin, 1),  # Sine function
        "Cos": (torch.cos, 1),  # Cosine function
        "Square": (torch.square, 1),  # Square function
        "Max": (torch.maximum, 2),  # Maximum function
        "Min": (torch.minimum, 2),  # Minimum function
        "Neg": (
            torch.negative,
            1,
        ),  # Unary negation function (i.e. negate the input value)
        "RSin": (radian_sin_torch, 1),  # Sine function
        "RCos": (radian_cos_torch, 1),  # Cosine function
        "Gaussian": (
            gaussian_torch,
            1,
        ),  # Residual function for handling negative values
        "Relu": (torch.relu, 1),  # Residual function for handling negative values
        "Abs": (torch.absolute, 1),  # Absolute value function
    }[p]
    return primitive
