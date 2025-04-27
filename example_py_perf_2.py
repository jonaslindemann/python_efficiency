#!/bin/env python3
# -*- coding: utf-8 -*-

import timeit
import random
import math
from collections import deque

import matplotlib.pyplot as plt

def setup_data(n):
    """Create test data for demonstrations"""
    return list(range(n)), {i: i for i in range(n)}, set(range(n))

def setup_words(n):
    """Create a list of words for string concatenation tests"""
    return ["word" + str(i) for i in range(n)]

def setup_dict(n):
    """Create a dictionary for repeated lookup tests"""
    items = list(range(n))
    prices = {i: random.uniform(1, 100) for i in items}
    quantities = {i: random.randint(1, 10) for i in items}
    return items, prices, quantities

def setup_valid_positions(n):
    """Create a list of valid positions for repeated function call tests"""
    valid_positions = {}
    n_x = int(math.sqrt(n))
    n_y = n_x
    for x in range(n_x):
        for y in range(n_y):
            valid_positions[(x, y)] = True  # Simulate valid positions
    return valid_positions

# Size of data structure

n_values = [10, 100, 1000, 10000, 100000]

growing_list_1_times = []
growing_list_2_times = []
string_concatenation1_times = []
string_concatenation2_times = []
dict_lookup1_times = []
dict_lookup2_times = []
test_repeated_func_call1_times = []
test_repeated_func_call2_times = []
test_comp_derived_values1_times = []
test_comp_derived_values2_times = []   
test_nested_red_calc1_times = []
test_nested_red_calc2_times = [] 


for n in n_values:

    print(f"Running tests for n = {n}")

    # Create data structures
    data_list, data_dict, data_set = setup_data(n)
    words = setup_words(n)
    items, prices, quantities = setup_dict(n)
    valid_positions = setup_valid_positions(n)

    # Elements to search for
    elements_to_find = [random.randint(0, n-1) for _ in range(1000)]
    element_not_present = n + 1

    def test_growing_lists_1():
        """Test growing lists"""
        result = []
        for i in range(n):
            result = result + [i]  # Creates a new list each iteration

    def test_growing_lists_2():
        """Test growing lists"""
        result = []
        for i in range(n):
            result.append(i)

    def test_string_concatenation1():
        """Test string concatenation"""
        text = ""
        for word in words:
            text = text + word + " "  # Creates new string each iteration

    def test_string_concatenation2():
        """Test string concatenation"""
        # Build a list then join once at the end
        word_list = []
        for word in words:
            word_list.append(word)
        text = " ".join(word_list)

    def test_dict_lookup1():
        """Test repeated dictionary lookup"""
        total = 0
        for item in items:
            # Lookup happens every iteration
            total += prices[item] * quantities[item]

    def test_dict_lookup2():
        """Test repeated dictionary lookup"""
        total = 0
        for item in items:
            # Cache lookups in local variables
            price = prices[item]
            quantity = quantities[item]
            total += price * quantity

    def is_valid_position(x, y):
        """Check if the position is valid"""
        return 0 <= x < n and 0 <= y < n
    
    def process_position(x, y):
        """Process the position"""
        # Simulate some processing
        return x * y
        
    def test_repeated_func_call1():
        """Test repeated function call"""

        n_x = int(math.sqrt(n))
        n_y = n_x

        for x in range(n_x):
            for y in range(n_y):
                # Expensive function called repeatedly with same inputs
                if is_valid_position(x, y):
                    process_position(x, y)

    def test_repeated_func_call2():
        """Test repeated function call"""

        n_x = int(math.sqrt(n))
        n_y = n_x

        # Use cached results
        for x in range(n_x):
            for y in range(n_y):
                if (x, y) in valid_positions:
                    process_position(x, y)

    def process(value):
        """Process data"""
        # Simulate some processing
        return value*value

    def test_comp_derived_values1():
        """Test derived values"""
        for i in range(len(data_list)):
            # len(data) called in every iteration
            s = f"Processing {i+1} of {len(data_list)}"
            process(data_list[i])

    def test_comp_derived_values2():
        """Test derived values"""
        data_length = len(data_list)  # Calculate once
        for i in range(data_length):
            s = f"Processing {i+1} of {data_length}"
            process(data_list[i])

    def test_nested_red_calc1():
        """Test nested repeated calculations"""
        n_x = int(math.sqrt(n))
        n_y = n_x

        total = 0
        for i in range(n_x):
            for j in range(n_y):
                # Math.pow is expensive and called n² times
                total += i * math.pow(j, 2)

    def test_nested_red_calc2():
        """Test nested repeated calculations"""
        n_x = int(math.sqrt(n))
        n_y = n_x

        total = 0
        for i in range(n_x):
            # Pre-compute the sum for the inner loop
            j_squared_sum = sum(j**2 for j in range(n_y))
            total += i * j_squared_sum

    def test_nested_red_calc3():
        """Test nested repeated calculations"""
        n_x = int(math.sqrt(n))
        n_y = n_x

        total = 0
        for i in range(n_x):
            # Pre-compute the sum for the inner loop
            j_squared_sum = sum(j**2 for j in range(n_y))
            total += i * j_squared_sum

    # Run benchmarks
    growing_list_1 = timeit.timeit(test_growing_lists_1, number=5)
    growing_list_2 = timeit.timeit(test_growing_lists_2, number=5)
    string_concatenation1 = timeit.timeit(test_string_concatenation1, number=5)
    string_concatenation2 = timeit.timeit(test_string_concatenation2, number=5)
    dict_lookup1 = timeit.timeit(test_dict_lookup1, number=5)
    dict_lookup2 = timeit.timeit(test_dict_lookup2, number=5)
    test_repeated_func_call1 = timeit.timeit(test_repeated_func_call1, number=5)
    test_repeated_func_call2 = timeit.timeit(test_repeated_func_call2, number=5)
    test_comp_derived_values1 = timeit.timeit(test_comp_derived_values1, number=5)
    test_comp_derived_values2 = timeit.timeit(test_comp_derived_values2, number=5)
    test_nested_red_calc1 = timeit.timeit(test_nested_red_calc1, number=5)
    test_nested_red_calc2 = timeit.timeit(test_nested_red_calc2, number=5)

    growing_list_1_times.append(growing_list_1)
    growing_list_2_times.append(growing_list_2)
    string_concatenation1_times.append(string_concatenation1)
    string_concatenation2_times.append(string_concatenation2)
    dict_lookup1_times.append(dict_lookup1)
    dict_lookup2_times.append(dict_lookup2)
    test_repeated_func_call1_times.append(test_repeated_func_call1)
    test_repeated_func_call2_times.append(test_repeated_func_call2)
    test_comp_derived_values1_times.append(test_comp_derived_values1)
    test_comp_derived_values2_times.append(test_comp_derived_values2)
    test_nested_red_calc1_times.append(test_nested_red_calc1)
    test_nested_red_calc2_times.append(test_nested_red_calc2)


# Plotting the results

plt.figure(figsize=(12, 8))
plt.plot(n_values, growing_list_1_times, label='+ operator', marker='o')
plt.plot(n_values, growing_list_2_times, label='append()', marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Growing lists time complexity')
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(n_values, string_concatenation1_times, label='+ operator', marker='o')
plt.plot(n_values, string_concatenation2_times, label='append()', marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('String concatenation time complexity')
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(n_values, string_concatenation1_times, label='repeated', marker='o')
plt.plot(n_values, string_concatenation2_times, label='cached', marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Repeated dictionary lookup time complexity')
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(n_values, test_repeated_func_call1_times, label='repeated', marker='o')
plt.plot(n_values, test_repeated_func_call2_times, label='cached', marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Repeated function call time complexity')
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(n_values, test_comp_derived_values1_times, label='repeated', marker='o')
plt.plot(n_values, test_comp_derived_values2_times, label='pre-calc', marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Computing the same derived values repeatedly')
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(n_values, test_nested_red_calc1_times, label='nested', marker='o')
plt.plot(n_values, test_nested_red_calc2_times, label='pre-comp', marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Nested loops with redundant calculations')
plt.legend()

plt.show()