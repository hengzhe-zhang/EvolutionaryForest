from collections import OrderedDict
from functools import wraps


def custom_lru_cache(maxsize=128, key_func=None, condition_func=None):
    cache = OrderedDict()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if condition_func and condition_func(*args, **kwargs):
                return func(*args, **kwargs)

            key = key_func(*args, **kwargs) if key_func else (args, frozenset(kwargs.items()))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return result

        return wrapper

    return decorator


def gp_key_func(gene, pset, data, **kwarg1):
    """
    Configurations are not change during the whole evolution process, and thus they do not need to be used as key
    """
    tree = str(gene)
    """
    data/pset: ensure compatible with modular GP
    """
    data.flags.writeable = False
    random_seed = kwarg1.get('random_seed', None)
    random_noise = kwarg1.get('random_noise', None)
    return_subtree_information = kwarg1.get('return_subtree_information', None)
    return (tree +
            ' ' + str(random_seed) +
            ' ' + str(random_noise) +
            ' ' + str(id(pset)) +
            ' ' + str(data.data.tobytes()) +
            ' ' + str(return_subtree_information))
