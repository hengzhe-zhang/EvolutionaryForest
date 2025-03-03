from collections import OrderedDict
from functools import wraps


def custom_lru_cache(maxsize=128, key_func=None, condition_func=None):
    cache = OrderedDict()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if condition_func and condition_func(*args, **kwargs):
                return func(*args, **kwargs)

            key = (
                key_func(*args, **kwargs)
                if key_func
                else (args, frozenset(kwargs.items()))
            )
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
