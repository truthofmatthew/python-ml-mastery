# Day 6: Optimizers and Learning Rate Scheduling in PyTorch

In this tutorial, we will delve into the world of optimizers and learning rate scheduling in PyTorch. Optimizers are algorithms or methods used to adjust the attributes of your neural network such as weights and learning rate in order to reduce the losses. Learning rate scheduling is a method to adjust the learning rate during training by reducing the learning rate according to a pre-defined schedule.

## Task 1: Implement an Optimizer to Train a Simple Model

Let's start by creating a simple linear regression model and use an optimizer to train it.

```python
import torch
import torch.nn as nn

# Define a simple linear model
model = nn.Linear(2, 1)

# Define a simple dataset
x = torch.randn(100, 2)
y = torch.randn(100, 1)

# Define a loss function
loss_fn = nn.MSELoss()

# Define an optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for i in range(1000):
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss = loss_fn(y_pred, y)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
```

In the above code, we first define a simple linear model and a dataset. We then define a mean squared error loss function and a stochastic gradient descent (SGD) optimizer. In the training loop, we perform a forward pass, compute the loss, zero the gradients, perform a backward pass, and finally update the weights using the optimizer.

## Task 2: Experiment with Different Optimizers

PyTorch provides several optimizers such as SGD, Adam, RMSProp, etc. Let's experiment with the Adam optimizer.

```python
# Define an Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for i in range(1000):
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss = loss_fn(y_pred, y)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
```

In the above code, we simply replace the SGD optimizer with the Adam optimizer. Adam is an adaptive learning rate optimization algorithm that's been designed specifically for training deep neural networks.

## Task 3: Apply Learning Rate Scheduling to Improve Training

Learning rate scheduling can be used to adjust the learning rate during training. PyTorch provides several learning rate schedulers such as StepLR, ExponentialLR, and ReduceLROnPlateau. Let's use the StepLR scheduler.

```python
# Define an optimizer and a scheduler
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Training loop
for i in range(1000):
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss = loss_fn(y_pred, y)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Step the scheduler
    scheduler.step()
```

In the above code, we define a StepLR scheduler that multiplies the learning rate by 0.1 every 100 epochs. In the training loop, we add a call to `scheduler.step()` after updating the weights.

In conclusion, optimizers and learning rate scheduling are crucial components of training neural networks in PyTorch. Different optimizers and learning rate strategies can have a significant impact on the performance of your model.