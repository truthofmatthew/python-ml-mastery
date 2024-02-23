# Hyperparameter Tuning in PyTorch

Hyperparameters are the variables that govern the training process and the topology of an ML model. These variables remain constant over the training process and directly impact the performance and efficiency of the model. In this tutorial, we will explore how to tune hyperparameters in PyTorch.

## Task 1: Explore the effects of learning rate on model training

The learning rate is a critical hyperparameter in the training of neural networks. It determines the step size at which the model learns. If the learning rate is too high, the model might overshoot the optimal point. If it's too low, the model might need too many iterations to converge to the best values. Let's see how different learning rates affect the training process.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 2),
    nn.Softmax(dim=1)
)

# Define a simple dataset
x = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Define a loss function
loss_fn = nn.CrossEntropyLoss()

# Try different learning rates
for lr in [0.1, 0.01, 0.001, 0.0001]:
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
    print(f'Learning rate: {lr}, Loss: {loss.item()}')
```

## Task 2: Use grid search or random search to find optimal hyperparameters

Grid search and random search are two common methods for hyperparameter tuning. Grid search is a brute-force exhaustive search paradigm where you specify a list of values for different hyperparameters, and the computer evaluates the model performance for each combination of these to find the best set.

Random search, on the other hand, selects random combinations of the hyperparameters to evaluate.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

# Define a simple model
model = MLPClassifier()

# Define the hyperparameters
hyperparameters = {
    'hidden_layer_sizes': [(10, 5), (5, 2)],
    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
    'max_iter': [100, 200, 300]
}

# Perform grid search
grid_search = GridSearchCV(model, hyperparameters, cv=5)
grid_search.fit(x.numpy(), y.numpy())
print(f'Best parameters: {grid_search.best_params_}')
```

## Task 3: Implement an experiment tracking tool to log your tuning process

Experiment tracking tools like TensorBoard, MLflow, or Weights & Biases can help you log and visualize your model training process, including the changes in loss and accuracy, the distribution of weights and biases, etc.

```python
from torch.utils.tensorboard import SummaryWriter

# Define a writer
writer = SummaryWriter()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()
    optimizer.step()

    # Log the loss
    writer.add_scalar('Loss', loss.item(), epoch)

# Close the writer
writer.close()
```

In this tutorial, we have explored how to tune hyperparameters in PyTorch. Remember, the goal of hyperparameter tuning is to find the optimal configuration of hyperparameters that minimizes the loss function and maximizes the performance of the model.