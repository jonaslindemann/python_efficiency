import timeit

# Option 1: Using a list comprehension
list_comp_code = """
squares = [x**2 for x in range(1000)]
"""

# Option 2: Using a traditional for loop
for_loop_code = """
squares = []
for x in range(1000):
    squares.append(x**2)
"""

# Running the benchmarks
list_comp_time = timeit.timeit(list_comp_code, number=10000)
for_loop_time = timeit.timeit(for_loop_code, number=10000)

print(f"List comprehension: {list_comp_time:.6f} seconds")
print(f"For loop: {for_loop_time:.6f} seconds")
print(f"List comprehension is {for_loop_time/list_comp_time:.2f}x faster")