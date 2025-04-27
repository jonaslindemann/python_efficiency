import numpy as np
import time
from numba import jit

# Define a computationally intensive function
def regular_function(size):
    """Standard Python implementation of a matrix operation."""
    result = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            result[i, j] = np.sin(i * j) + np.cos(i + j)
    return result

# The same function with Numba JIT compilation
@jit(nopython=True)  # nopython=True forces compilation without falling back to Python
def numba_function(size):
    """Numba-accelerated implementation of the same matrix operation."""
    result = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            result[i, j] = np.sin(i * j) + np.cos(i + j)
    return result

# Benchmark both functions
def benchmark():
    size = 1000
    
    # Warm-up Numba (first call includes compilation time)
    _ = numba_function(10)
    
    # Time the regular function
    start = time.time()
    regular_result = regular_function(size)
    regular_time = time.time() - start
    print(f"Regular function time: {regular_time:.4f} seconds")
    
    # Time the Numba function
    start = time.time()
    numba_result = numba_function(size)
    numba_time = time.time() - start
    print(f"Numba function time: {numba_time:.4f} seconds")
    
    # Calculate speedup
    speedup = regular_time / numba_time
    print(f"Speedup with Numba: {speedup:.2f}x")
    
    # Verify that results are the same
    print(f"Results match: {np.allclose(regular_result, numba_result)}")

if __name__ == "__main__":
    benchmark()