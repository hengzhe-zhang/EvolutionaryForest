import numpy as np
from deap.gp import PrimitiveSet


def get_pset(data):
    pset = PrimitiveSet("MAIN", data.shape[1])
    pset.addPrimitive(add, 2)
    pset.addPrimitive(subtract, 2)
    pset.addPrimitive(multiply, 2)
    pset.addPrimitive(aq, 2)
    pset.addPrimitive(sqrt, 1)
    pset.addPrimitive(abs_log, 1)
    pset.addPrimitive(np.abs, 1)
    pset.addPrimitive(np.square, 1)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(sin_pi, 1)
    pset.addPrimitive(cos_pi, 1)
    pset.addPrimitive(np.negative, 1)
    return pset


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def sqrt(x):
    return np.sqrt(np.abs(x))


def abs_log(x):
    return np.log(np.abs(x) + 1)


def aq(x, y):
    return x / np.sqrt(1 + y**2)


def sin_pi(x):
    return np.sin(np.pi * x)


def cos_pi(x):
    return np.cos(np.pi * x)


def multiply(x, y):
    return x * y
