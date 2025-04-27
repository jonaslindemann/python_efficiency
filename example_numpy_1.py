import numpy as np
import time
import timeit
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def pure_python_matrix_multiplication(A, B):
    """Matrix multiplication using pure Python lists."""
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

def numpy_matrix_multiplication(A, B):
    """Matrix multiplication using NumPy arrays."""
    return np.dot(A, B)

def benchmark_comparison(sizes):
    """
    Compare performance between pure Python and NumPy for matrix multiplication
    across different matrix sizes.
    """
    python_times = []
    numpy_times = []
    
    for size in sizes:
        # Create random matrices
        py_matrix_A = [[np.random.random() for _ in range(size)] for _ in range(size)]
        py_matrix_B = [[np.random.random() for _ in range(size)] for _ in range(size)]
        
        # Convert to NumPy arrays
        np_matrix_A = np.array(py_matrix_A)
        np_matrix_B = np.array(py_matrix_B)
        
        # Time pure Python implementation
        start = time.time()
        _ = pure_python_matrix_multiplication(py_matrix_A, py_matrix_B)
        end = time.time()
        python_time = end - start
        python_times.append(python_time)
        print(f"Pure Python - Size {size}x{size}: {python_time:.6f} seconds")
        
        # Time NumPy implementation
        start = time.time()
        _ = numpy_matrix_multiplication(np_matrix_A, np_matrix_B)
        end = time.time()
        numpy_time = end - start
        numpy_times.append(numpy_time)
        print(f"NumPy - Size {size}x{size}: {numpy_time:.6f} seconds")
        
        # Calculate speedup
        speedup = python_time / numpy_time
        print(f"NumPy is {speedup:.2f}x faster\n")
    
    return python_times, numpy_times

def plot_comparison(sizes, python_times, numpy_times):
    """Plot the performance comparison between Pure Python and NumPy."""
    plt.figure(figsize=(12, 10))
    
    # Comparison plot with logarithmic scale
    plt.subplot(2, 1, 1)
    plt.plot(sizes, python_times, 'o-', label='Pure Python')
    plt.plot(sizes, numpy_times, 's-', label='NumPy')
    plt.yscale('log')
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.title('Performance Comparison: Pure Python vs NumPy Matrix Multiplication')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    # Speedup plot
    plt.subplot(2, 1, 2)
    speedups = [p/n for p, n in zip(python_times, numpy_times)]
    plt.bar(sizes, speedups, color='green', alpha=0.7)
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Speedup Factor (times faster)')
    plt.title('NumPy Speedup Factor Compared to Pure Python')
    
    # Add text labels above each bar
    for i, v in enumerate(speedups):
        plt.text(sizes[i], v + 5, f"{v:.1f}x", ha='center')
    
    plt.tight_layout()
    plt.savefig('python_vs_numpy_performance.png', dpi=300)
    plt.show()
    
    return speedups

def memory_usage_comparison():
    """Compare memory usage between Python lists and NumPy arrays."""
    import sys
    
    sizes = [10, 100, 1000, 10000]
    python_mem = []
    numpy_mem = []
    
    for size in sizes:
        # Python list
        py_list = [[0.0 for _ in range(size)] for _ in range(size)]
        python_size = sys.getsizeof(py_list) + sum(sys.getsizeof(row) for row in py_list)
        python_mem.append(python_size / (1024 * 1024))  # Convert to MB
        
        # NumPy array
        np_array = np.zeros((size, size))
        numpy_size = np_array.nbytes / (1024 * 1024)  # Convert to MB
        numpy_mem.append(numpy_size)
        
        print(f"Size {size}x{size}:")
        print(f"  Python List: {python_size / (1024 * 1024):.2f} MB")
        print(f"  NumPy Array: {numpy_size:.2f} MB")
    
    # Plot memory comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, python_mem, 'o-', label='Python List')
    plt.plot(sizes, numpy_mem, 's-', label='NumPy Array')
    plt.xlabel('Matrix Size (n×n)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage: Python Lists vs NumPy Arrays')
    plt.grid(True, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('python_vs_numpy_memory.png', dpi=300)
    plt.show()

def element_wise_operations_comparison():
    """Compare element-wise operations between Python lists and NumPy arrays."""
    sizes = [100, 200, 500, 1000]
    python_times = []
    numpy_times = []
    
    for size in sizes:
        # Create data
        python_list = [i * 0.01 for i in range(size)]
        numpy_array = np.array(python_list)
        
        # Python list element-wise multiplication
        python_time = timeit.timeit(lambda: [x * 2 for x in python_list])
        python_times.append(python_time)
        
        # NumPy array element-wise multiplication
        numpy_time = timeit.timeit(lambda: numpy_array * 2)
        numpy_times.append(numpy_time)
        
        print(f"Size {size}:")
        print(f"  Python List: {python_time:.6f} seconds")
        print(f"  NumPy Array: {numpy_time:.6f} seconds")
        print(f"  NumPy is {python_time/numpy_time:.2f}x faster\n")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, python_times, 'o-', label='Python List')
    plt.plot(sizes, numpy_times, 's-', label='NumPy Array')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds) - Log Scale')
    plt.title('Element-wise Operations: Python Lists vs NumPy Arrays')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('python_vs_numpy_elementwise.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    print("=== Matrix Multiplication Performance Comparison ===")
    # Use smaller sizes for faster execution
    #matrix_sizes = [200, 400, 600]
    #python_times, numpy_times = benchmark_comparison(matrix_sizes)
    #speedups = plot_comparison(matrix_sizes, python_times, numpy_times)
    
    #print("\n=== Memory Usage Comparison ===")
    #memory_usage_comparison()
    
    print("\n=== Element-wise Operations Comparison ===")
    element_wise_operations_comparison()