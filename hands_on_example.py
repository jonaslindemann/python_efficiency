#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random

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


def main():
    import time
    import math

    # Generate large data
    size = 100000  # Adjust size as needed
    data = generate_large_data(size)

    # Process data with a threshold
    threshold = 50

    # Measure execution time
    start_time = time.time()
    results = process_data(data, threshold)
    end_time = time.time()

    print(f"Processed {len(results)} items in {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    main()