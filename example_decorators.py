import time
import functools

def timing_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper

@timing_decorator
def slow_operation(n):
    """A deliberately slow function for demonstration"""
    total = 0
    for i in range(n):
        total += i
    return total

slow_operation(10000000)