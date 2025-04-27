import numpy as np
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

def pure_python_matrix_multiply(A, B):
    """
    Multiply two matrices using pure Python.
    A and B are 2D lists representing matrices.
    """
    # Initialize result matrix with zeros
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])
    
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    # Perform matrix multiplication
    for i in range(rows_A):
        print(i)
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
                
    return C

def numpy_matrix_multiply(A, B):
    """
    Multiply two matrices using NumPy.
    A and B are numpy arrays.
    """
    return np.matmul(A, B)

def compute_euclidean_distances_python(X):
    """
    Compute pairwise Euclidean distances between points in pure Python.
    X is a 2D list where each row is a point.
    """
    n = len(X)
    distances = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate Euclidean distance between points i and j
                dist = 0
                for k in range(len(X[0])):
                    dist += (X[i][k] - X[j][k]) ** 2
                distances[i][j] = dist ** 0.5
                
    return distances

def compute_euclidean_distances_numpy(X):
    """
    Compute pairwise Euclidean distances between points using NumPy.
    X is a numpy array where each row is a point.
    """
    # Calculate squared distances using broadcasting
    sq_dist = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    
    # Take square root to get Euclidean distances
    return np.sqrt(sq_dist)

def run_benchmark(sizes):
    """Run benchmark comparing NumPy vs Pure Python for various operations."""
    results = {
        'size': sizes,
        'matrix_pure_python': [],
        'matrix_numpy': [],
        'speedup_matrix': [],
        'distance_pure_python': [],
        'distance_numpy': [],
        'speedup_distance': []
    }
    
    for size in sizes:
        print(f"Testing size {size}...")
        
        # Create random matrices
        A_list = [[np.random.random() for _ in range(size)] for _ in range(size)]
        B_list = [[np.random.random() for _ in range(size)] for _ in range(size)]
        
        A_np = np.array(A_list)
        B_np = np.array(B_list)
        
        if size<=1000:
            # Matrix multiplication benchmarking
            start_time = time.time()
            _ = pure_python_matrix_multiply(A_list, B_list)
            pure_python_time = time.time() - start_time
            results['matrix_pure_python'].append(pure_python_time)
        
        start_time = time.time()
        _ = numpy_matrix_multiply(A_np, B_np)
        numpy_time = time.time() - start_time
        results['matrix_numpy'].append(numpy_time)
        
        results['speedup_matrix'].append(pure_python_time / numpy_time)
        
        # For larger sizes, we'll only test Euclidean distances on a subset of points
        points_size = size  # Limit to 100 points max for distance calculation
        
        # Create random points
        points_list = [[np.random.random() for _ in range(10)] for _ in range(points_size)]
        points_np = np.array(points_list)

        if size <= 1000:
            # Euclidean distance benchmarking
            start_time = time.time()
            _ = compute_euclidean_distances_python(points_list)
            pure_python_time = time.time() - start_time
            results['distance_pure_python'].append(pure_python_time)
        
        start_time = time.time()
        _ = compute_euclidean_distances_numpy(points_np)
        numpy_time = time.time() - start_time
        results['distance_numpy'].append(numpy_time)

        if size <= 1000:
            results['speedup_distance'].append(pure_python_time / numpy_time)        
    
    return results

def plot_results(results):
    """Plot the benchmark results."""
    plt.figure(figsize=(14, 7))
    
    # Plot execution times
    plt.subplot(1, 2, 1)
    plt.plot(results['size'], results['matrix_pure_python'], 'o-', label='Pure Python - Matrix Mult.')
    plt.plot(results['size'], results['matrix_numpy'], 'o-', label='NumPy - Matrix Mult.')
    plt.plot(results['size'], results['distance_pure_python'], 's-', label='Pure Python - Distances')
    plt.plot(results['size'], results['distance_numpy'], 's-', label='NumPy - Distances')
    plt.xlabel('Size')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot speedup
    plt.subplot(1, 2, 2)
    plt.plot(results['size'], results['speedup_matrix'], 'o-', label='Matrix Multiplication')
    plt.plot(results['size'], results['speedup_distance'], 's-', label='Euclidean Distances')
    plt.xlabel('Size')
    plt.ylabel('Speedup (times faster)')
    plt.title('NumPy Speedup over Pure Python')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('numpy_vs_python_performance.png')
    plt.show()

def print_results_table(results):
    """Print the benchmark results as a table."""
    table_data = []
    for i, size in enumerate(results['size']):
        table_data.append([
            size,
            f"{results['matrix_pure_python'][i]:.4f}",
            f"{results['matrix_numpy'][i]:.4f}",
            f"{results['speedup_matrix'][i]:.2f}x",
            f"{results['distance_pure_python'][i]:.4f}",
            f"{results['distance_numpy'][i]:.4f}",
            f"{results['speedup_distance'][i]:.2f}x"
        ])
    
    headers = [
        "Size", 
        "Matrix (Python)", 
        "Matrix (NumPy)", 
        "Speedup", 
        "Distances (Python)", 
        "Distances (NumPy)", 
        "Speedup"
    ]
    
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def main():
    # Run benchmark for increasing sizes
    sizes = [100, 1000, 10000]  # Adjust based on your computer's performance
    results = run_benchmark(sizes)
    
    # Print results table
    print_results_table(results)
    
    # Plot results
    plot_results(results)
    
    # Demonstrate NumPy's broadcasting capabilities
    print("\nDemonstrating NumPy's broadcasting capabilities:")
    
    # Create a 3x3 matrix
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("\nOriginal matrix A:")
    print(A)
    
    # Add a scalar to each element
    print("\nA + 10 (add 10 to each element):")
    print(A + 10)
    
    # Multiply each row by a different scalar
    row_factors = np.array([10, 100, 1000]).reshape(3, 1)
    print("\nMultiplying each row by [10, 100, 1000]:")
    print(A * row_factors)
    
    # Multiple operations combined
    print("\nComplex operations in one line: 2*A^2 + 3*A + 5")
    print(2 * A**2 + 3 * A + 5)

if __name__ == "__main__":
    main()