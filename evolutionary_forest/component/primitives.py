import numpy as np
from hdfe import Groupby
from scipy.stats import mode

threshold = 1e-6


def protect_divide(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > threshold, np.divide(x1, x2), 1.)


def analytical_quotient(x1, x2):
    return x1 / np.sqrt(1 + (x2 ** 2))


def individual_to_tuple(ind):
    arr = []
    for x in ind.gene:
        arr.append(tuple(a.name for a in x))
    return tuple(sorted(arr))


def protect_loge(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > threshold, np.log(np.abs(x1)), 0.)


def protect_sqrt(a):
    return np.sqrt(np.abs(a))


def np_max(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return max(x1, x2)
    x1, x2 = shape_wrapper(x1, x2)
    return np.max([x1, x2], axis=0)


def np_min(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return min(x1, x2)
    x1, x2 = shape_wrapper(x1, x2)
    return np.min([x1, x2], axis=0)


def shape_wrapper(x1, x2):
    if np.isscalar(x1):
        return np.full(x2.shape[0], x1), x2
    elif np.isscalar(x2):
        return x1, np.full(x1.shape[0], x2)
    else:
        return x1, x2


def group_sum(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return x2
    x1, x2 = shape_wrapper(x1, x2)
    return Groupby(x1).apply(np.sum, x2, broadcast=True)


def group_mean(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return x2
    x1, x2 = shape_wrapper(x1, x2)
    return Groupby(x1).apply(np.mean, x2, broadcast=True)


def group_count(x1):
    if np.isscalar(x1):
        return x1
    return Groupby(x1).apply(len, x1, broadcast=True)


def group_min(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return x2
    x1, x2 = shape_wrapper(x1, x2)
    return Groupby(x1).apply(np.min, x2, broadcast=True)


def group_max(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return x2
    x1, x2 = shape_wrapper(x1, x2)
    return Groupby(x1).apply(np.max, x2, broadcast=True)


def group_mode(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return x2
    x1, x2 = shape_wrapper(x1, x2)
    return Groupby(x1).apply(lambda x: mode(x).mode[0], x2, broadcast=True)


def identical_boolean(x):
    return x


def same_boolean(x):
    return x


def identical_categorical(x):
    return x


def same_categorical(x):
    return x


def identical_numerical(x):
    return x


def same_numerical(x):
    return x


def cross_product(x1, x2):
    if np.isscalar(x1) or np.isscalar(x2):
        return x1
    x = np.stack([x1, x2]).T
    return np.unique(x, axis=0, return_inverse=True)[1]


def add_basic_operators(pset):
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(analytical_quotient, 2)


def np_wrapper(func):
    def simple_func(x1, x2):
        if np.isscalar(x1) and np.isscalar(x2):
            return func(x1, x2).astype(int)
        x1, x2 = shape_wrapper(x1, x2)
        return func(x1, x2).astype(int)

    simple_func.__name__ = f'np_{func.__name__}'
    return simple_func


def np_bit_wrapper(func):
    def simple_func(x1, x2):
        if np.isscalar(x1) and np.isscalar(x2):
            return x1
        x1, x2 = shape_wrapper(x1, x2)
        if x1.dtype == int and x2.dtype == int:
            return func(x1, x2)
        else:
            return x1

    simple_func.__name__ = f'np_{func.__name__}'
    return simple_func


def add_logical_operators(pset):
    pset.addPrimitive(np_bit_wrapper(np.bitwise_and), 2)
    pset.addPrimitive(np_bit_wrapper(np.bitwise_or), 2)
    pset.addPrimitive(np_bit_wrapper(np.bitwise_xor), 2)


def add_relation_operators(pset):
    pset.addPrimitive(np_wrapper(np.greater), 2)
    pset.addPrimitive(np_wrapper(np.less), 2)
