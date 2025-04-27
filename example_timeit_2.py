import timeit

def list_comp():
    squares = [x**2 for x in range(1000)]
    return squares

def for_loop():
    squares = []
    for x in range(1000):
        squares.append(x**2)
    return squares

# Running the benchmarks
list_comp_time = timeit.timeit(list_comp, number=10000)
for_loop_time = timeit.timeit(for_loop, number=10000)

print(f"List comprehension: {list_comp_time:.6f} seconds")
print(f"For loop: {for_loop_time:.6f} seconds")
print(f"List comprehension is {for_loop_time/list_comp_time:.2f}x faster")