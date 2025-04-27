#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import numba as nb
import time

from functools import lru_cache


def generate_large_data(size):
    """Generate a large list of random integers."""
    data = [random.randint(1, 100) for _ in range(size)]
    return data

def process_data(data, threshold):
    results = []
    for i in range(len(data)):
        value = data[i]
        if value > threshold:
            transformed = 0
            for j in range(100):
                transformed += value * math.sin(j)
            results = results + [transformed]
    return results

import math

def process_data_optimized(data, threshold):
    # Pre-calculate the sum of sine values (constant for all data points)
    sin_sum = sum(math.sin(j) for j in range(100))
    
    # Use list comprehension for cleaner, faster filtering and transformation
    results = [value * sin_sum for value in data if value > threshold]
    
    return results

@lru_cache(maxsize=128)
def calculate_sin_sum(value):
    return value * sum(math.sin(j) for j in range(100))

def process_data_optimized_lru_cache(data, threshold):
    return [calculate_sin_sum(value) for value in data if value > threshold]


def process_data_numpy(data, threshold):
    # Convert input to numpy array if it's not already
    data_array = np.asarray(data)
    
    # Create mask for values above threshold
    mask = data_array > threshold
    
    # Filter data using mask
    filtered_data = data_array[mask]
    
    # Pre-calculate sine values for all j values (0-99)
    sin_values = np.sin(np.arange(100))
    
    # Multiply each filtered value by the sum of sine values
    return filtered_data * np.sum(sin_values)    

@nb.jit(nopython=True)
def process_data_numba(data, threshold):
    results = []
    # Pre-calculate sum of sines
    sin_sum = 0
    for j in range(100):
        sin_sum += math.sin(j)
        
    for i in nb.prange(len(data)):
        value = data[i]
        if value > threshold:
            transformed = value * sin_sum
            results.append(transformed)
    return results


def main():

    # Generate large data

    print("Generating large data...")
    size = 200000  # Adjust size as needed
    data = generate_large_data(size)

    # Process data with a threshold
    threshold = 50

    print("Processing data unoptimized...")
    start_time = time.time()
    results = process_data(data, threshold)
    end_time = time.time()

    t_unoptimized = end_time - start_time

    print("Processing data optimized (Python)...")
    start_time = time.time()
    results = process_data_optimized(data, threshold)
    end_time = time.time()

    t_optimized = end_time - start_time

    print("Processing data optimized (LRU Cache)...")
    start_time = time.time()
    results = process_data_optimized(data, threshold)
    end_time = time.time()

    t_optimized_lru = end_time - start_time

    print("Processing data optimized (NumPy)...")
    arr = np.array(data)
    start_time = time.time()
    results = process_data_numpy(data, threshold)
    end_time = time.time()

    t_optimized_numpy = end_time - start_time

    print("Processing data optimized (Numba)...")
    start_time = time.time()
    results = process_data_numba(data, threshold)
    end_time = time.time()

    t_optimized_numba = end_time - start_time

    print(f"Unoptimized: Processed {len(results)} items in {t_unoptimized:.4f} seconds.")
    print(f"Optimized: Processed {len(results)} items in {t_optimized:.4f} seconds.")
    print(f"Optimized with LRU Cache: Processed {len(results)} items in {t_optimized_lru:.4f} seconds.")
    print(f"Optimized with NumPy: Processed {len(results)} items in {t_optimized_numpy:.4f} seconds.")
    print(f"Optimized with Numba: Processed {len(results)} items in {t_optimized_numba:.4f} seconds.")
    print(f"Speedup: {t_unoptimized / t_optimized:.2f}x")
    print(f"Speedup with LRU Cache: {t_optimized / t_optimized_lru:.2f}x")

    print(f"Speedup with NumPy: {t_unoptimized / t_optimized_numpy:.2f}x")
    print(f"Speedup with Numba: {t_unoptimized / t_optimized_numba:.2f}x")

if __name__ == "__main__":
    main()