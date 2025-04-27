# Writing efficient Python Code

This repository contains a collection of Python code snippets and examples that demonstrate various techniques for writing efficient and optimized Python code. The examples are used in the workshop "Writing Efficient Python Code" for the COMPUTE Spring Meeting.

## Cloning the repository

To clone the repository, you can use the following command:

```bash
git clone https://github.com/jonaslindemann/python_efficiency
```

## Installing required packages

To install the required packages, you can use the following command:

```bash
cd python_efficiency
```

Then, create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml -n python-eff-env
```

This will create a new conda environment with the name `python-eff-env` and install all the required packages listed in the `environment.yml` file.

After the environment is created, you can activate it using the following command:

```bash
conda activate python-eff-env
```

## PyPy example

To run the PyPy example, you need to create a separate conda environment with PyPy installed. You can do this by running the following command:

```bash
conda create -n pypy-env pypy numpy matplotlib
```

Then, activate the PyPy environment:

```bash
conda activate pypy-env
```

You can then run the PyPy example using the following command:

```bash
pypy example_pypy.py
```
