from sympy.core import cache


def clear_sympy_cache():
    """
    Clears the sympy cache to ensure that the cache does not affect the benchmark results.
    """
    cache.clear_cache()


def warm_up_function(func, *args, **kwargs):
    """
    Runs the function a few times to ensure it is 'warmed up' and any initialization overhead is reduced.
    """
    for _ in range(5):
        func(*args, **kwargs)
