# Day 1: Setting Up Your Environment for PyTorch

## Introduction

To start your journey into the world of Python and machine learning, the first step is to set up your development environment. This tutorial will guide you through the process of installing PyTorch, a powerful open-source machine learning library for Python. 

## Learning Objective

By the end of this tutorial, you will be able to install PyTorch and set up a development environment. You will also learn to verify your PyTorch installation by running a sample code snippet and explore basic PyTorch operations.

## Task 1: Install PyTorch following official guidelines

PyTorch can be installed via pip or conda. The official PyTorch website provides a convenient installation command generator. Visit [PyTorch - Get Started](https://pytorch.org/get-started/locally/) and select your preferences.

For example, to install PyTorch on a Windows machine with pip, without CUDA, you would use the following command:

```bash
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Task 2: Verify PyTorch installation by running a sample code snippet

After installing PyTorch, you can verify the installation by running a simple script that prints the PyTorch version. Open your Python interpreter and type:

```python
import torch
print(torch.__version__)
```

If PyTorch is installed correctly, this will print the version of PyTorch that you installed.

## Task 3: Explore basic PyTorch operations (create tensors, addition, multiplication)

PyTorch operations are similar to NumPy but with GPU support. Let's explore some basic operations:

### Creating Tensors

Tensors are a type of data structure used in linear algebra, and like vectors and matrices, you can calculate arithmetic operations with tensors.

```python
# Create a tensor
x = torch.tensor([1, 2, 3])
print(x)
```

### Addition

You can perform addition between tensors.

```python
# Create tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Add tensors
z = x + y
print(z)
```

### Multiplication

Multiplication can be performed in two ways - elementwise multiplication or dot product.

```python
# Elementwise multiplication
z = x * y
print(z)

# Dot product
z = torch.dot(x, y)
print(z)
```

This concludes the first day of your journey into Python and machine learning with PyTorch. You have successfully set up your environment and performed basic operations in PyTorch. Keep practicing and exploring more functionalities of PyTorch.