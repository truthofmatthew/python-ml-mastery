# PyTorch Autograd for Automatic Differentiation

In this tutorial, we will explore PyTorch's `autograd` feature, which allows for automatic computation of gradients. This is a key component in training neural networks, as it enables the optimization of model parameters.

## Task 1: Create tensors with `requires_grad=True` and perform operations

To start, we need to create a tensor with `requires_grad=True`. This tells PyTorch to track all operations on the tensor, so it can automatically compute gradients.

```python
import torch

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

Now, let's perform some operations on the tensor:

```python
y = x + 2
z = y * y * 3
out = z.mean()

print(y)
print(z)
print(out)
```

## Task 2: Calculate gradients using the `.backward()` method

PyTorch computes gradients by calling the `.backward()` method on a tensor. This computes the gradient of the tensor with respect to some scalar value.

```python
# Compute gradients
out.backward()

# Print gradients d(out)/dx
print(x.grad)
```

## Task 3: Explore the role of gradients in training neural networks

Gradients play a crucial role in the training of neural networks. During training, we want to optimize the model's parameters so that the loss function is minimized. Gradients point in the direction of steepest ascent, so by moving in the opposite direction (i.e., performing gradient descent), we can iteratively find the parameters that minimize the loss.

Here's a simple example of how this might look in PyTorch:

```python
# Create random tensors for weights and biases
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Set learning rate and number of epochs
learning_rate = 0.01
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    # Forward pass: compute predicted y using operations on tensors
    y_pred = w * x + b
    
    # Compute and print loss
    loss = (y_pred - y).pow(2).mean()
    print(f'Epoch {epoch}, Loss {loss.item()}')
    
    # Use autograd to compute the backward pass, calculating gradients
    loss.backward()
    
    # Update weights and biases using gradient descent
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Manually zero the gradients after updating weights
    w.grad.zero_()
    b.grad.zero_()
```

In this example, we use the gradients computed by `autograd` to perform the gradient descent step. After each update, we need to manually zero the gradients to prevent them from accumulating.

This tutorial has provided a brief introduction to PyTorch's `autograd` feature. By automatically computing gradients, `autograd` simplifies the process of training neural networks and allows us to focus on designing our models.