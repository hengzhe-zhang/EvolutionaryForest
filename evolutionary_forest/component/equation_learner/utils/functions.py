"""Functions for use with symbolic regression.

These functions encapsulate multiple implementations (sympy, Tensorflow, numpy) of a particular function so that the
functions can be used in multiple contexts."""

import torch
# import tensorflow as tf
import numpy as np
import sympy as sp


class BaseFunction:
    """Abstract class for primitive functions"""

    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        """Sympy implementation"""
        return None

    def torch(self, x):
        """No need for base function"""
        return None

    def tf(self, x):
        """Automatically convert sympy to TensorFlow"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'tensorflow')(x)

    def np(self, x):
        """Automatically convert sympy to numpy"""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)

    def name(self, x):
        return str(self.sp)


class Constant(BaseFunction):
    def torch(self, x):
        return torch.ones_like(x)

    def tf(self, x):
        return tf.ones_like(x)

    def sp(self, x):
        return 1

    def np(self, x):
        return np.ones_like


class Identity(BaseFunction):
    def torch(self, x):
        return x / self.norm  # ??

    def tf(self, x):
        return tf.identity(x) / self.norm

    def sp(self, x):
        return x / self.norm

    def np(self, x):
        return np.array(x) / self.norm


class Square(BaseFunction):
    def torch(self, x):
        return torch.square(x) / self.norm

    def tf(self, x):
        return tf.square(x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

    def np(self, x):
        return np.square(x) / self.norm


class Pow(BaseFunction):
    def __init__(self, power, norm=1):
        BaseFunction.__init__(self, norm=norm)
        self.power = power

    def torch(self, x):
        return torch.pow(x, self.power) / self.norm

    def sp(self, x):
        return x ** self.power / self.norm

    def tf(self, x):
        return tf.pow(x, self.power) / self.norm


class Sin(BaseFunction):
    def torch(self, x):
        return torch.sin(x * 2 * 2 * np.pi) / self.norm

    def sp(self, x):
        return sp.sin(x * 2 * 2 * np.pi) / self.norm


class Sigmoid(BaseFunction):
    def torch(self, x):
        return torch.sigmoid(x) / self.norm

    # def tf(self, x):
    #     return tf.sigmoid(x) / self.norm

    def sp(self, x):
        return 1 / (1 + sp.exp(-20 * x)) / self.norm

    def np(self, x):
        return 1 / (1 + np.exp(-20 * x)) / self.norm

    def name(self, x):
        return "sigmoid(x)"


class Exp(BaseFunction):
    def __init__(self, norm=np.e):
        super().__init__(norm)

    # ?? why the minus 1
    def torch(self, x):
        return (torch.exp(x) - 1) / self.norm

    def sp(self, x):
        return (sp.exp(x) - 1) / self.norm


class Log(BaseFunction):
    def torch(self, x):
        return torch.log(torch.abs(x)) / self.norm

    def sp(self, x):
        return sp.log(sp.Abs(x)) / self.norm


class BaseFunction2:
    """Abstract class for primitive functions with 2 inputs"""

    def __init__(self, norm=1.):
        self.norm = norm

    def sp(self, x, y):
        """Sympy implementation"""
        return None

    def torch(self, x, y):
        return None

    def tf(self, x, y):
        """Automatically convert sympy to TensorFlow"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'tensorflow')(x, y)

    def np(self, x, y):
        """Automatically convert sympy to numpy"""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)

    def name(self, x, y):
        return str(self.sp)


class Product(BaseFunction2):
    def __init__(self, norm=0.1):
        super().__init__(norm=norm)

    def torch(self, x, y):
        return x * y / self.norm

    def sp(self, x, y):
        return x * y / self.norm


def count_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction):
            i += 1
        elif isinstance(func, BaseFunction2):
            i += 2
    return i


def count_double(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction2):
            i += 1
    return i


# default_func = [
#     Constant(),
#     Constant(),
#     Identity(),
#     Identity(),
#     Square(),
#     Square(),
#     Sin(),
#     Sigmoid(),
# ]

default_func = [
    *[Constant()] * 2,
    *[Identity()] * 4,
    *[Square()] * 4,
    *[Sin()] * 2,
    *[Exp()] * 2,
    *[Sigmoid()] * 2,
    *[Product()] * 2,
]
