# Day 2: Understanding Tensors in PyTorch

Tensors are the primary data structure used in PyTorch and are similar to arrays in NumPy. They are used to encode the inputs and outputs of a model, as well as the model’s parameters.

## Task 1: Create tensors of different types and sizes

To create a tensor, you can use the `torch.tensor()` function, which accepts a data array as input. Here is an example of creating a tensor from a list:

```python
import torch

# Create a tensor from a list
x = torch.tensor([1, 2, 3, 4, 5])
print(x)
```

You can also create a tensor of a specific size with uninitialized memory using `torch.empty()`, or a tensor filled with zeros using `torch.zeros()`, or ones using `torch.ones()`:

```python
# Create an empty tensor
x = torch.empty(5, 3)
print(x)

# Create a tensor filled with zeros
x = torch.zeros(5, 3)
print(x)

# Create a tensor filled with ones
x = torch.ones(5, 3)
print(x)
```

## Task 2: Perform element-wise operations and reduction operations

Element-wise operations are operations that are performed on corresponding elements of tensors. Here is an example of element-wise addition:

```python
# Create two tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Perform element-wise addition
z = x + y
print(z)
```

Reduction operations reduce the number of elements in a tensor. Here is an example of sum reduction:

```python
# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5])

# Perform sum reduction
sum_x = torch.sum(x)
print(sum_x)
```

## Task 3: Understand the concept of broadcasting in tensor operations

Broadcasting is a powerful mechanism that allows PyTorch to work with arrays of different shapes when performing arithmetic operations. Here is an example:

```python
# Create a tensor
x = torch.tensor([1, 2, 3])

# Create a scalar
y = torch.tensor(2)

# Perform element-wise multiplication with broadcasting
z = x * y
print(z)
```

In this example, the scalar `y` is broadcasted to the shape of `x` before the multiplication operation. The result is a tensor where each element of `x` is multiplied by `y`.