#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# First, install the memory_profiler package
# pip install memory_profiler

from memory_profiler import profile
import numpy as np
import random
import time

# Use the @profile decorator to monitor memory usage of specific functions
@profile
def create_large_list(size):
    """Create a large list of integers"""
    print("Creating large list...")
    large_list = [random.randint(1, 100) for _ in range(size)]
    time.sleep(1)  # Pause to clearly see memory usage
    return large_list

@profile
def create_large_numpy_array(size):
    """Create a large numpy array"""
    print("Creating large numpy array...")
    large_array = np.random.randint(1, 100, size=size)
    time.sleep(1)  # Pause to clearly see memory usage
    return large_array

@profile
def process_data_inefficient(size):
    """Process data with inefficient memory usage"""
    data = create_large_list(size)
    
    # Create multiple copies of the data
    processed = []
    for i in range(10):
        # This creates multiple copies of large chunks of data
        processed.append(data[int(i * size/10):int((i+1) * size/10)] * 2)
    
    time.sleep(1)  # Pause to clearly see memory usage
    return processed

@profile
def process_data_efficient(size):
    """Process data with more efficient memory usage"""
    data = create_large_numpy_array(size)
    
    # Process in place with numpy (more memory efficient)
    processed = []
    for i in range(10):
        # Using views instead of copies
        section = data[int(i * size/10):int((i+1) * size/10)]
        # Avoid creating unnecessary intermediate copies
        processed.append(section * 2)
    
    time.sleep(1)  # Pause to clearly see memory usage
    return processed

if __name__ == "__main__":
    # Run with a moderate size to see differences without using too much memory
    data_size = 100000  # 10 million elements
    
    print("\n=== Testing inefficient implementation ===")
    result1 = process_data_inefficient(data_size)
    
    print("\n=== Testing efficient implementation ===")
    result2 = process_data_efficient(data_size)

# python -m memory_profiler example5.py