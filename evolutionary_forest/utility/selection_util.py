import functools
import inspect


def has_status_arg(func):
    # Unwrap if it's a functools.partial
    while isinstance(func, functools.partial):
        func = func.func

    sig = inspect.signature(func)
    return "status" in sig.parameters
