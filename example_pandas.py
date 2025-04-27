import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import random
import os

# Create sample data file
def create_sample_data(filename, num_rows=100000):
    """Create a sample CSV file with sales data for testing."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Generate realistic sales data
    headers = ['date', 'customer_id', 'product', 'category', 'quantity', 'unit_price', 'total']
    products = [
        ('Laptop', 'Electronics', 899.99), 
        ('Smartphone', 'Electronics', 699.99),
        ('Headphones', 'Electronics', 159.99),
        ('T-shirt', 'Clothing', 24.99),
        ('Jeans', 'Clothing', 49.99),
        ('Sneakers', 'Footwear', 89.99),
        ('Coffee Maker', 'Home', 129.99),
        ('Blender', 'Home', 79.99),
        ('Book', 'Books', 19.99),
        ('Tablet', 'Electronics', 349.99)
    ]
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(num_rows):
            date = datetime(2023, random.randint(1, 12), random.randint(1, 28)).strftime('%Y-%m-%d')
            customer_id = random.randint(1000, 9999)
            product, category, unit_price = random.choice(products)
            quantity = random.randint(1, 5)
            total = quantity * unit_price
            
            writer.writerow([date, customer_id, product, category, quantity, unit_price, total])
    
    print(f"Created sample data file with {num_rows} rows at {filename}")

def pure_python_analysis(filename):
    """Perform data analysis using pure Python."""
    start_time = time.time()
    
    # Read CSV file
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'date': row['date'],
                'customer_id': int(row['customer_id']),
                'product': row['product'],
                'category': row['category'],
                'quantity': int(row['quantity']),
                'unit_price': float(row['unit_price']),
                'total': float(row['total'])
            })
    
    read_time = time.time()
    print(f"Pure Python - Reading data: {read_time - start_time:.4f} seconds")
    
    # 1. Group by category and calculate total sales
    category_sales = {}
    for row in data:
        category = row['category']
        total = row['total']
        if category in category_sales:
            category_sales[category] += total
        else:
            category_sales[category] = total
    
    group_time = time.time()
    print(f"Pure Python - Group by category: {group_time - read_time:.4f} seconds")
    
    # 2. Find the top 3 selling products
    product_sales = {}
    for row in data:
        product = row['product']
        total = row['total']
        if product in product_sales:
            product_sales[product] += total
        else:
            product_sales[product] = total
    
    top_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:3]
    
    top_products_time = time.time()
    print(f"Pure Python - Find top products: {top_products_time - group_time:.4f} seconds")
    
    # 3. Calculate monthly sales
    monthly_sales = {}
    for row in data:
        month = row['date'][:7]  # Extract YYYY-MM
        total = row['total']
        if month in monthly_sales:
            monthly_sales[month] += total
        else:
            monthly_sales[month] = total
    
    monthly_time = time.time()
    print(f"Pure Python - Monthly aggregation: {monthly_time - top_products_time:.4f} seconds")
    
    # 4. Filter for high-value transactions (>= $500)
    high_value = [row for row in data if row['total'] >= 500]
    
    filter_time = time.time()
    print(f"Pure Python - Filtering: {filter_time - monthly_time:.4f} seconds")
    
    # 5. Calculate average order value by customer
    customer_orders = {}
    customer_totals = {}
    for row in data:
        cust_id = row['customer_id']
        total = row['total']
        if cust_id in customer_orders:
            customer_orders[cust_id] += 1
            customer_totals[cust_id] += total
        else:
            customer_orders[cust_id] = 1
            customer_totals[cust_id] = total
    
    avg_order_value = {cust_id: customer_totals[cust_id] / customer_orders[cust_id] for cust_id in customer_orders}
    
    avg_time = time.time()
    print(f"Pure Python - Customer analytics: {avg_time - filter_time:.4f} seconds")
    
    total_time = avg_time - start_time
    print(f"Pure Python - Total execution time: {total_time:.4f} seconds\n")
    
    return {
        'read_time': read_time - start_time,
        'group_time': group_time - read_time,
        'top_products_time': top_products_time - group_time,
        'monthly_time': monthly_time - top_products_time,
        'filter_time': filter_time - monthly_time,
        'avg_time': avg_time - filter_time,
        'total_time': total_time,
        'results': {
            'category_sales': category_sales,
            'top_products': top_products,
            'monthly_sales': monthly_sales,
            'high_value_count': len(high_value),
            'avg_order_sample': list(avg_order_value.items())[:3]  # Sample of first 3
        }
    }

def pandas_analysis(filename):
    """Perform the same data analysis using Pandas."""
    start_time = time.time()
    
    # Read CSV file
    df = pd.read_csv(filename)
    
    read_time = time.time()
    print(f"Pandas - Reading data: {read_time - start_time:.4f} seconds")
    
    # 1. Group by category and calculate total sales
    category_sales = df.groupby('category')['total'].sum().to_dict()
    
    group_time = time.time()
    print(f"Pandas - Group by category: {group_time - read_time:.4f} seconds")
    
    # 2. Find the top 3 selling products
    top_products = df.groupby('product')['total'].sum().nlargest(3).reset_index().values.tolist()
    top_products = [(item[0], item[1]) for item in top_products]  # Convert to the same format as pure Python
    
    top_products_time = time.time()
    print(f"Pandas - Find top products: {top_products_time - group_time:.4f} seconds")
    
    # 3. Calculate monthly sales
    df['month'] = df['date'].str[:7]  # Extract YYYY-MM
    monthly_sales = df.groupby('month')['total'].sum().to_dict()
    
    monthly_time = time.time()
    print(f"Pandas - Monthly aggregation: {monthly_time - top_products_time:.4f} seconds")
    
    # 4. Filter for high-value transactions (>= $500)
    high_value = df[df['total'] >= 500]
    
    filter_time = time.time()
    print(f"Pandas - Filtering: {filter_time - monthly_time:.4f} seconds")
    
    # 5. Calculate average order value by customer
    avg_order_value = df.groupby('customer_id')['total'].mean().to_dict()
    
    avg_time = time.time()
    print(f"Pandas - Customer analytics: {avg_time - filter_time:.4f} seconds")
    
    total_time = avg_time - start_time
    print(f"Pandas - Total execution time: {total_time:.4f} seconds\n")
    
    return {
        'read_time': read_time - start_time,
        'group_time': group_time - read_time,
        'top_products_time': top_products_time - group_time,
        'monthly_time': monthly_time - top_products_time,
        'filter_time': filter_time - monthly_time,
        'avg_time': avg_time - filter_time,
        'total_time': total_time,
        'results': {
            'category_sales': category_sales,
            'top_products': top_products,
            'monthly_sales': monthly_sales,
            'high_value_count': len(high_value),
            'avg_order_sample': list(avg_order_value.items())[:3]  # Sample of first 3
        }
    }

def data_join_comparison(num_rows=50000):
    """Compare joining and merging data between pure Python and Pandas."""
    # Create two sample datasets
    print("Creating sample datasets for join comparison...")
    
    # Sample data for customers
    customers_data = []
    for i in range(1, 1001):
        customers_data.append({
            'customer_id': i,
            'name': f'Customer {i}',
            'email': f'customer{i}@example.com',
            'signup_date': f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}'
        })
    
    # Sample data for orders
    orders_data = []
    for i in range(num_rows):
        orders_data.append({
            'order_id': i + 1,
            'customer_id': random.randint(1, 1000),  # Random customer from our list
            'order_date': f'2023-{random.randint(1,12):02d}-{random.randint(1,28):02d}',
            'order_total': round(random.uniform(10, 1000), 2)
        })
    
    # Pure Python join
    start_time = time.time()
    
    # Build lookup dictionary for customers
    customer_lookup = {customer['customer_id']: customer for customer in customers_data}
    
    # Join orders with customer information
    joined_data = []
    for order in orders_data:
        customer_id = order['customer_id']
        if customer_id in customer_lookup:
            joined_record = {
                'order_id': order['order_id'],
                'customer_id': customer_id,
                'customer_name': customer_lookup[customer_id]['name'],
                'customer_email': customer_lookup[customer_id]['email'],
                'order_date': order['order_date'],
                'order_total': order['order_total']
            }
            joined_data.append(joined_record)
    
    python_time = time.time() - start_time
    print(f"Pure Python - Join operation: {python_time:.4f} seconds")
    
    # Convert to pandas DataFrames
    customers_df = pd.DataFrame(customers_data)
    orders_df = pd.DataFrame(orders_data)
    
    # Pandas join
    start_time = time.time()
    
    # Merge DataFrames
    joined_df = orders_df.merge(customers_df, on='customer_id', how='inner')
    
    pandas_time = time.time() - start_time
    print(f"Pandas - Join operation: {pandas_time:.4f} seconds")
    
    return {
        'python_time': python_time,
        'pandas_time': pandas_time,
        'speedup': python_time / pandas_time
    }

def plot_comparison_results(py_results, pd_results):
    """Plot the performance comparison results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Operation time comparison
    operations = ['Reading', 'Grouping', 'Top Products', 'Monthly Agg', 'Filtering', 'Customer Avg']
    py_times = [py_results['read_time'], py_results['group_time'], 
                 py_results['top_products_time'], py_results['monthly_time'], 
                 py_results['filter_time'], py_results['avg_time']]
    
    pd_times = [pd_results['read_time'], pd_results['group_time'], 
                 pd_results['top_products_time'], pd_results['monthly_time'], 
                 pd_results['filter_time'], pd_results['avg_time']]
    
    x = np.arange(len(operations))
    width = 0.35
    
    ax1.bar(x - width/2, py_times, width, label='Pure Python')
    ax1.bar(x + width/2, pd_times, width, label='Pandas')
    
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Operation Performance: Python vs Pandas')
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, rotation=45, ha='right')
    ax1.legend()
    
    # Total time and speedup
    total_times = [py_results['total_time'], pd_results['total_time']]
    speedup = py_results['total_time'] / pd_results['total_time']
    
    ax2.bar(['Pure Python', 'Pandas'], total_times, color=['#1f77b4', '#ff7f0e'])
    ax2.text(1, pd_results['total_time'] + 0.5, f"{speedup:.2f}x faster", 
             ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Total Execution Time')
    
    plt.tight_layout()
    plt.savefig('python_vs_pandas_performance.png', dpi=300)
    plt.close()
    
    # Plot join comparison
    join_results = data_join_comparison()
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Pure Python', 'Pandas'], [join_results['python_time'], join_results['pandas_time']], 
            color=['#1f77b4', '#ff7f0e'])
    plt.text(1, join_results['pandas_time'] + 0.2, f"{join_results['speedup']:.2f}x faster", 
             ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Time (seconds)')
    plt.title('Data Join Performance: Pure Python vs Pandas')
    plt.tight_layout()
    plt.savefig('python_vs_pandas_join.png', dpi=300)
    plt.close()

def pivot_table_comparison():
    """Compare creating a pivot table in pure Python vs Pandas."""
    # Create sample data
    np.random.seed(42)
    data = []
    regions = ['North', 'South', 'East', 'West']
    products = ['A', 'B', 'C', 'D']
    
    for _ in range(10000):
        region = random.choice(regions)
        product = random.choice(products)
        sales = round(random.uniform(100, 1000), 2)
        month = random.randint(1, 12)
        data.append({
            'region': region,
            'product': product,
            'month': month,
            'sales': sales
        })
    
    # Pure Python pivot table
    start_time = time.time()
    
    # Build the pivot table: Region × Product, sum of sales
    pivot_data = {}
    for region in regions:
        pivot_data[region] = {}
        for product in products:
            pivot_data[region][product] = 0
    
    # Fill with data
    for row in data:
        region = row['region']
        product = row['product']
        sales = row['sales']
        pivot_data[region][product] += sales
    
    python_time = time.time() - start_time
    print(f"Pure Python - Pivot table: {python_time:.4f} seconds")
    
    # Pandas pivot table
    df = pd.DataFrame(data)
    
    start_time = time.time()
    pivot_df = df.pivot_table(index='region', columns='product', values='sales', aggfunc='sum')
    pandas_time = time.time() - start_time
    print(f"Pandas - Pivot table: {pandas_time:.4f} seconds")
    
    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.bar(['Pure Python', 'Pandas'], [python_time, pandas_time], color=['#1f77b4', '#ff7f0e'])
    plt.text(1, pandas_time + 0.05, f"{python_time / pandas_time:.2f}x faster", 
             ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Time (seconds)')
    plt.title('Pivot Table Creation: Pure Python vs Pandas')
    plt.tight_layout()
    plt.savefig('python_vs_pandas_pivot.png', dpi=300)
    plt.close()
    
    return {
        'python_time': python_time,
        'pandas_time': pandas_time,
        'speedup': python_time / pandas_time
    }

def code_complexity_comparison():
    """Compare code complexity for data operations."""
    # Data reading
    py_read = """
# Pure Python CSV reading
data = []
with open('sales.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append({
            'date': row['date'],
            'customer_id': int(row['customer_id']),
            'product': row['product'],
            'category': row['category'],
            'quantity': int(row['quantity']),
            'unit_price': float(row['unit_price']),
            'total': float(row['total'])
        })
    """
    
    pd_read = """
# Pandas CSV reading
df = pd.read_csv('sales.csv')
    """
    
    # Filtering
    py_filter = """
# Pure Python filtering
high_value = []
for row in data:
    if row['total'] >= 500:
        high_value.append(row)
    """
    
    pd_filter = """
# Pandas filtering
high_value = df[df['total'] >= 500]
    """
    
    # Grouping
    py_group = """
# Pure Python grouping
category_sales = {}
for row in data:
    category = row['category']
    total = row['total']
    if category in category_sales:
        category_sales[category] += total
    else:
        category_sales[category] = total
    """
    
    pd_group = """
# Pandas grouping
category_sales = df.groupby('category')['total'].sum()
    """
    
    # Create a chart
    operations = ['Data Reading', 'Filtering', 'Grouping']
    py_lines = [len(py_read.strip().split('\n')), len(py_filter.strip().split('\n')), len(py_group.strip().split('\n'))]
    pd_lines = [len(pd_read.strip().split('\n')), len(pd_filter.strip().split('\n')), len(pd_group.strip().split('\n'))]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(operations))
    width = 0.35
    
    plt.bar(x - width/2, py_lines, width, label='Pure Python')
    plt.bar(x + width/2, pd_lines, width, label='Pandas')
    
    plt.ylabel('Lines of Code')
    plt.title('Code Complexity: Pure Python vs Pandas')
    plt.xticks(x, operations)
    
    # Add percentage reduction
    for i, (py, pd) in enumerate(zip(py_lines, pd_lines)):
        reduction = (py - pd) / py * 100
        plt.text(i, max(py, pd) + 0.5, f"{reduction:.0f}% reduction", ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('python_vs_pandas_complexity.png', dpi=300)
    plt.close()
    
    return {
        'py_lines': py_lines,
        'pd_lines': pd_lines,
        'reduction': [(py - pd) / py * 100 for py, pd in zip(py_lines, pd_lines)]
    }

if __name__ == "__main__":
    # Generate sample data
    data_file = "data/sales_data.csv"
    create_sample_data(data_file)
    
    # Run the analyses
    print("=== Running Pure Python Analysis ===")
    py_results = pure_python_analysis(data_file)
    
    print("=== Running Pandas Analysis ===")
    pd_results = pandas_analysis(data_file)
    
    # Create comparison plots
    print("\n=== Creating Performance Comparisons ===")
    plot_comparison_results(py_results, pd_results)
    
    print("\n=== Pivot Table Comparison ===")
    pivot_results = pivot_table_comparison()
    
    print("\n=== Code Complexity Comparison ===")
    complexity_results = code_complexity_comparison()
    
    print("\nAll comparisons completed! Check the generated PNG files for visualization results.")