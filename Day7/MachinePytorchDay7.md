# Implementing a Training Loop in PyTorch

In this tutorial, we will implement a training loop in PyTorch. The training loop is the core part of the learning process where the model iteratively learns from the data. We will also monitor the training progress by printing out the loss at intervals and evaluate the model on a validation set.

## Task 1: Implement a Training Loop Including Forward and Backward Passes

Let's start by defining a simple model. We will use a simple linear regression model for this tutorial.

```python
import torch
import torch.nn as nn

# Define a simple linear regression model
model = nn.Linear(10, 1)
```

Next, we need to define a loss function and an optimizer. We will use Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent (SGD) as the optimizer.

```python
# Define loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Now, we can implement the training loop. In each iteration of the training loop, we perform a forward pass, calculate the loss, perform a backward pass to compute gradients, and update the model parameters.

```python
# Training loop
for epoch in range(100):  # Number of epochs
    for batch in train_loader:  # For each batch in the training data
        # Forward pass
        outputs = model(batch['input'])
        loss = loss_fn(outputs, batch['target'])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Task 2: Monitor Training Progress by Printing Out Loss at Intervals

To monitor the training progress, we can print out the loss at regular intervals. We will modify the training loop to print the loss every 10 epochs.

```python
# Training loop
for epoch in range(100):  # Number of epochs
    for batch in train_loader:  # For each batch in the training data
        # Forward pass
        outputs = model(batch['input'])
        loss = loss_fn(outputs, batch['target'])

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
```

## Task 3: Evaluate the Model on a Validation Set

After training the model, we need to evaluate it on a validation set to check its performance. We can do this by running the model on the validation data and calculating the loss.

```python
# Switch to evaluation mode
model.eval()

with torch.no_grad():  # No need to calculate gradients
    total_loss = 0
    for batch in val_loader:  # For each batch in the validation data
        # Forward pass
        outputs = model(batch['input'])
        loss = loss_fn(outputs, batch['target'])
        total_loss += loss.item()

# Print average validation loss
print(f'Validation Loss: {total_loss / len(val_loader)}')
```

In this tutorial, we have implemented a complete training loop in PyTorch, monitored the training progress by printing out the loss at intervals, and evaluated the model on a validation set.