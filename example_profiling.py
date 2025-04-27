#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cProfile
import pstats
import io
from pstats import SortKey

# Function we want to profile
def analyze_data(data_size):
    # Create some sample data
    data = list(range(data_size))
    
    # Some potentially expensive operations
    result1 = process_data_method1(data)
    result2 = process_data_method2(data)
    result3 = process_data_method3(data)
    
    return result1, result2, result3

def process_data_method1(data):
    # Simple iteration
    result = 0
    for i in data:
        result += i * i
    return result

def process_data_method2(data):
    # List comprehension
    return sum([i * i for i in data])

def process_data_method3(data):
    # Generator expression
    return sum(i * i for i in data)

# Create a profile object
profiler = cProfile.Profile()

# Start profiling
profiler.enable()

# Run the function we want to profile
analyze_data(1000000)

# Stop profiling
profiler.disable()

# Print sorted stats to console
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats(10)  # Print top 10 time-consuming functions
print(s.getvalue())

# Save results to file for more detailed analysis later
ps.dump_stats('analysis_results.prof')