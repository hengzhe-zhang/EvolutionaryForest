import operator
import time
from functools import reduce

import numpy as np
import pandas as pd
import torch
from hdfe import Groupby
from numba import jit
from scipy.stats import mode

threshold = 1e-6


def protected_division(x1, *x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        x2 = reduce(operator.mul, x2)
        return np.where(np.abs(x2) > threshold, np.divide(x1, x2), 1.)


def protected_division_torch(x1, *x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        x2 = reduce(operator.mul, x2)
        return torch.where(torch.abs(x2) > threshold, torch.divide(x1, x2), 1.)


def protected_log(x1, x2):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > threshold, np.emath.logn(np.abs(x2), np.abs(x1)), 0.)


def protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > threshold, np.divide(1, x1), 0.)


def analytical_quotient(x1, x2):
    return x1 / np.sqrt(1 + (x2 ** 2))


def analytical_quotient_torch(x1, x2):
    return x1 / torch.sqrt(1 + (x2 ** 2))


def individual_to_tuple(ind):
    arr = []
    for x in ind.gene:
        arr.append(tuple(a.name for a in x))
    if hasattr(ind, 'base_model'):
        arr.append((ind.base_model,))
    return tuple(sorted(arr))


def protect_loge(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > threshold, np.log(np.abs(x1)), 0.)


def analytical_log(x):
    return np.log(np.sqrt(1 + x ** 2))


def analytical_log10(x):
    return np.log10(np.sqrt(1 + x ** 2))


def cube(x):
    return np.power(x, 3)


def protect_sqrt(a):
    return np.sqrt(np.abs(a))


def protect_sqrt_torch(a):
    return torch.sqrt(torch.abs(a))


def shape_wrapper(x1, x2):
    if np.isscalar(x1):
        return np.full(x2.shape[0], x1), x2
    elif np.isscalar(x2):
        return x1, np.full(x1.shape[0], x2)
    else:
        return x1, x2


def shape_wrapper_plus(*args):
    # new shape wrapper
    length = max([np.size(x) for x in args])
    result = []
    for x in args:
        if np.size(x) != length:
            result.append(np.full(length, x))
        else:
            result.append(x)
    return tuple(result)


def groupby_generator(function, argument_count=2):
    # support group_by operators based on Pandas
    def groupby_function(*args):
        if all([np.isscalar(x) for x in args]):
            return args[-1]
        args = shape_wrapper_plus(*args)
        frame = pd.DataFrame(np.array(args).T, columns=[f'C{x}' for x in range(len(args))])
        calculation = frame.groupby([f'C{x}' for x in range(len(args) - 1)])[f'C{len(args) - 1}']. \
            transform(function).to_numpy()
        return calculation

    groupby_function.__name__ = f'groupby_function_{function}_{argument_count}'
    return groupby_function


@jit(nopython=True)
def groupby_mean(keys, values):
    unique_keys = np.unique(keys)
    means = np.zeros_like(unique_keys, dtype=np.float64)
    counts = np.zeros_like(unique_keys, dtype=np.int64)

    # Calculate the sum and counts for each group
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        means[idx] += values[i]
        counts[idx] += 1

    # Divide the sum by counts to get the mean
    means /= counts

    # Map the means back to the original keys
    output_array = np.zeros_like(values, dtype=np.float64)
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        output_array[i] = means[idx]

    return output_array


@jit(nopython=True)
def groupby_max(keys, values):
    unique_keys = np.unique(keys)
    max_values = np.full_like(unique_keys, -np.inf, dtype=np.float64)  # Initialize with negative infinity

    # Find the max value for each group
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        if values[i] > max_values[idx]:
            max_values[idx] = values[i]

    # Map the max values back to the original keys
    output_array = np.zeros_like(values, dtype=np.float64)
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        output_array[i] = max_values[idx]

    return output_array


@jit(nopython=True)
def groupby_min(keys, values):
    unique_keys = np.unique(keys)
    min_values = np.full_like(unique_keys, np.inf, dtype=np.float64)  # Initialize with positive infinity

    # Find the min value for each group
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        if values[i] < min_values[idx]:
            min_values[idx] = values[i]

    # Map the min values back to the original keys
    output_array = np.zeros_like(values, dtype=np.float64)
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        output_array[i] = min_values[idx]

    return output_array


@jit(nopython=True)
def groupby_count(keys):
    unique_keys = np.unique(keys)
    counts = np.zeros_like(unique_keys, dtype=np.int64)

    # Count the occurrences for each key
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        counts[idx] += 1

    # Map the counts back to the original keys
    output_array = np.zeros_like(keys, dtype=np.int64)
    for i in range(len(keys)):
        idx = np.searchsorted(unique_keys, keys[i])
        output_array[i] = counts[idx]

    return output_array


def group_mode(x1, x2):
    if np.isscalar(x1) and np.isscalar(x2):
        return x2
    x1, x2 = shape_wrapper(x1, x2)
    return Groupby(x1).apply(lambda x: mode(x).mode[0], x2, broadcast=True)


def identical_boolean(x):
    return x.astype(float)


def same_boolean(x):
    return x


def identical_categorical(x):
    return x


def same_categorical(x):
    return x


def identical_numerical(x):
    return x


def identical_random_seed(x):
    return x


def same_numerical(x):
    return x


# Currently, the most tricky issue is that there may exist some labels which are not existing in training data
# Thus, the encoding process might be affected if such a feature exists
def cross_product(x1, x2):
    if np.isscalar(x1) or np.isscalar(x2):
        return x1
    x1, x2 = shape_wrapper(x1, x2)
    return x1 + ',' + x2


def if_function(x1, x2, x3):
    if np.isscalar(x1) and np.isscalar(x2) and np.isscalar(x3):
        return x1
    return np.where(x1 > 0, x2, x3)


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
            return func(x1, x2)
        x1, x2 = shape_wrapper(x1, x2)
        if x1.dtype == int and x2.dtype == int:
            return func(x1, x2)
        else:
            raise Exception

    simple_func.__name__ = f'np_{func.__name__}'
    return simple_func


def add_logical_operators(pset):
    pset.addPrimitive(np_bit_wrapper(np.bitwise_and), 2)
    pset.addPrimitive(np_bit_wrapper(np.bitwise_or), 2)
    pset.addPrimitive(np_bit_wrapper(np.bitwise_xor), 2)


def add_relation_operators(pset):
    pset.addPrimitive(np_wrapper(np.greater), 2)
    pset.addPrimitive(np_wrapper(np.less), 2)


def cross_feature(x1, x2):
    vf = np.vectorize(hash)
    return vf(x1) + vf(x2)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def residual(x):
    return x - np.round(x)


def np_mean(a, b):
    return np.mean([a, b], axis=0)


def greater_or_equal_than(a, b):
    return np.where(a >= b, 1, 0)


def less_or_equal_than(a, b):
    return np.where(a <= b, 1, 0)


def greater_or_equal_than_quadruple_a(a, b, c, d):
    return np.where(a >= b, c, d)


def greater_or_equal_than_quadruple_b(a, b, c, d):
    return np.where(a >= b, c + d, c - d)


def greater_or_equal_than_quadruple_c(a, b, c, d):
    return np.where(a >= b, c + d, c * d)


def less_than_quadruple_a(a, b, c, d):
    return np.where(a < b, c, d)


def less_than_quadruple_b(a, b, c, d):
    return np.where(a < b, c + d, c - d)


def less_than_quadruple_c(a, b, c, d):
    return np.where(a < b, c + d, c * d)


def and_a(a, b):
    return np.where((a >= 0) & (b >= 0), 1, 0)


def and_b(a, b):
    return np.where((a < 0) & (b < 0), 1, 0)


def or_a(a, b):
    return np.where((a >= 0) | (b >= 0), 1, 0)


def or_b(a, b):
    return np.where((a < 0) | (b < 0), 1, 0)


def greater_or_equal_than_double_a(a, b):
    return np.where(a >= b, a, b)


def less_than_double_a(a, b):
    return np.where(a < b, a, b)


def greater_or_equal_than_double_b(a, b):
    return np.where(a >= b, a + b, a - b)


def less_than_double_b(a, b):
    return np.where(a < b, a + b, a - b)


def greater_or_equal_than_double_c(a, b):
    return np.where(a >= b, a + b, a * b)


def less_than_double_c(a, b):
    return np.where(a < b, a + b, a * b)


def add_extend_operators(pset, addtional=False, groupby_operators=True,
                         triple_groupby_operators=False):
    ### Must pay attention that allowing more aruguments will increase the model size
    def scipy_mode(x):
        return mode(x).mode[0]

    # Basic features
    pset.addPrimitive(protect_sqrt, 1)
    pset.addPrimitive(analytical_log, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)

    # GroupBy features
    if groupby_operators:
        pset.addPrimitive(groupby_generator('mean'), 2)
        pset.addPrimitive(groupby_generator('max'), 2)
        pset.addPrimitive(groupby_generator('min'), 2)

    if triple_groupby_operators:
        pset.addPrimitive(groupby_generator('mean', 3), 3)
        pset.addPrimitive(groupby_generator('max', 3), 3)
        pset.addPrimitive(groupby_generator('min', 3), 3)

    if addtional:
        pset.addPrimitive(np.arctan, 1)
        pset.addPrimitive(np.tanh, 1)
        pset.addPrimitive(np.cbrt, 1)
        pset.addPrimitive(np.square, 1)
        pset.addPrimitive(np.negative, 1)


def groupby_operator_speed_test():
    a = np.random.randint(0, 5)
    b = np.random.rand(10000)
    mean_groupby = groupby_generator('mean')
    median_groupby = groupby_generator('median')
    max_groupby = groupby_generator('max')
    min_groupby = groupby_generator('min')
    sum_groupby = groupby_generator('sum')
    unique_groupby = groupby_generator('nunique')
    for g in [mean_groupby, median_groupby, max_groupby, min_groupby,
              sum_groupby, unique_groupby]:
        st = time.time()
        result = g(a, b)
        print(result)
        print('time', time.time() - st)


if __name__ == '__main__':
    # print(protected_division(1, 2, 2))
    keys = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    values = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    result = groupby_mean(keys, values)
    print(result)  # Output: [10. 25. 25. 50. 50. 50.]
    assert np.array_equal(result, [10, 25, 25, 50, 50, 50])

    keys = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    values = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    result = groupby_max(keys, values)
    print(result)  # Output: [10. 30. 30. 60. 60. 60.]
    assert np.array_equal(result, [10, 30, 30, 60, 60, 60, ])

    keys = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    values = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    result = groupby_min(keys, values)
    print(result)  # Output: [10. 20. 20. 40. 40. 40.]
    assert np.array_equal(result, [10, 20, 20, 40, 40, 40, ])

    keys = np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)
    result = groupby_count(keys)
    print(result)  # Output: [1 2 2 3 3 3]
    assert np.array_equal(result, [1, 2, 2, 3, 3, 3])
