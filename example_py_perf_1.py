#!/bin/env python3
# -*- coding: utf-8 -*-

import timeit
import random
from collections import deque

import matplotlib.pyplot as plt

def setup_data(n):
    """Create test data for demonstrations"""
    return list(range(n)), {i: i for i in range(n)}, set(range(n))

# Size of data structure

n_values = [10, 100, 1000, 10000]
list_times = []
dict_times = []
set_times = []

list_insert_times = []
deque_insert_times = []

for n in n_values:

    # Create data structures
    data_list, data_dict, data_set = setup_data(n)

    # Elements to search for
    elements_to_find = [random.randint(0, n-1) for _ in range(1000)]
    element_not_present = n + 1

    def test_list_membership():
        """Test membership checking in list"""
        count = 0
        for element in elements_to_find:
            if element in data_list:  # O(n) operation
                count += 1
        return count

    def test_dict_membership():
        """Test membership checking in dictionary"""
        count = 0
        for element in elements_to_find:
            if element in data_dict:  # O(1) operation
                count += 1
        return count

    def test_set_membership():
        """Test membership checking in set"""
        count = 0
        for element in elements_to_find:
            if element in data_set:  # O(1) operation
                count += 1
        return count

    def test_list_insertion():
        """Test inserting at beginning of list"""
        test_list = data_list.copy()
        for i in range(1000):
            test_list.insert(0, i)  # O(n) operation

    def test_deque_insertion():
        """Test inserting at beginning of deque"""
        test_deque = deque(data_list)
        for i in range(1000):
            test_deque.appendleft(i)  # O(1) operation

    # Run benchmarks
    list_time = timeit.timeit(test_list_membership, number=5)
    dict_time = timeit.timeit(test_dict_membership, number=5)
    set_time = timeit.timeit(test_set_membership, number=5)

    list_insert_time = timeit.timeit(test_list_insertion, number=5)
    deque_insert_time = timeit.timeit(test_deque_insertion, number=5)

    print(f"Membership testing (1000 elements in {n} items):")
    print(f"List: {list_time:.6f} seconds")
    print(f"Dict: {dict_time:.6f} seconds")
    print(f"Set: {set_time:.6f} seconds")
    print(f"List is {list_time/set_time:.1f}x slower than Set")

    print("\nInsertion at beginning (1000 operations):")
    print(f"List insert(0): {list_insert_time:.6f} seconds")
    print(f"Deque appendleft: {deque_insert_time:.6f} seconds")
    print(f"List is {list_insert_time/deque_insert_time:.1f}x slower than Deque")

    list_times.append(list_time)
    dict_times.append(dict_time)
    set_times.append(set_time)

    list_insert_times.append(list_insert_time)
    deque_insert_times.append(deque_insert_time)


# Plotting the results

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(n_values, list_times, label='List', marker='o')
plt.plot(n_values, dict_times, label='Dict', marker='o')
plt.plot(n_values, set_times, label='Set', marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Membership Testing Time Complexity')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(n_values, list_insert_times, label='List Insert(0)', marker='o')
plt.plot(n_values, deque_insert_times, label='Deque Appendleft', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of elements (log scale)')
plt.ylabel('Time (seconds, log scale)')
plt.title('Insertion Time Complexity')
plt.legend()
plt.tight_layout()
plt.show()