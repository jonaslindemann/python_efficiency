import time
import math

def compute_intensive_task(n):
    """
    A computationally intensive function that performs many iterations.
    This type of code typically benefits from PyPy's JIT compilation.
    """
    result = 0
    for i in range(n):
        result += math.sin(i) * math.cos(i)
        for j in range(100):
            result += math.sin(j) * math.cos(j) / (i + 1)
    return result

def benchmark():
    iterations = 10000
    
    # Time the function
    start = time.time()
    result = compute_intensive_task(iterations)
    elapsed = time.time() - start
    
    print(f"Result: {result}")
    print(f"Execution time: {elapsed:.4f} seconds")
    print(f"Using: {'PyPy' if 'pypy' in sys.version.lower() else 'CPython'}")

if __name__ == "__main__":
    import sys
    print(f"Python version: {sys.version}")
    benchmark()

"""
HOW TO RUN THIS EXAMPLE:

1. Save this file as "example11.py"

2. Run with standard CPython:
   $ python example11.py

3. Run with PyPy (assuming PyPy is installed):
   $ pypy example11.py

4. Compare the execution times
"""