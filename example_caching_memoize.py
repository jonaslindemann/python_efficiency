import functools
import timeit

from matplotlib.pylab import f

def memoize(func):
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the function arguments
        key = str(args) + str(kwargs)
        
        # Return cached result if available
        if key in cache:
            return cache[key]
        
        # Calculate result and store in cache
        result = func(*args, **kwargs)
        cache[key] = result
        return result
        
    return wrapper

@memoize
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_no_decorator(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


# Without memoization: O(2^n) complexity
# With memoization: O(n) complexity

t_cached = timeit.timeit(lambda: fibonacci(35))
t_std = timeit.timeit(lambda: fibonacci_no_decorator(35))

print(f"Time with memoization: {t_cached:.6f} seconds")
print(f"Time without memoization: {t_std:.6f} seconds")
print(f"Speedup: {t_std/t_cached:.2f}x")