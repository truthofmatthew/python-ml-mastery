# Day 18: Advanced PyTorch Techniques for Model Improvement

In this tutorial, we will explore advanced techniques to enhance your models in PyTorch. These techniques include dropout, batch normalization, and regularization. By the end of this tutorial, you will be able to implement these techniques in your own models.

## Task 1: Apply Dropout to Your Network

Dropout is a regularization technique that prevents overfitting by randomly setting a fraction of input units to 0 at each update during training. This helps to prevent units from co-adapting too much.

Let's apply dropout to a simple neural network:

```python
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

In the above code, we have added a dropout layer with a dropout probability of 0.5. During training, approximately half of the neurons in the preceding layer will be turned off randomly at each step.

## Task 2: Use Batch Normalization in a CNN

Batch normalization is a technique that can improve the learning rate of a neural network. It normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.

Let's apply batch normalization to a convolutional neural network (CNN):

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x
```

In the above code, we have added batch normalization layers after each convolutional layer. These layers will normalize the activations of the previous layer at each batch.

## Task 3: Experiment with Different Regularization Techniques

Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function. The two most common types of regularization are L1 and L2 regularization.

In PyTorch, you can add L2 regularization by adding a `weight_decay` parameter to the optimizer:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)
```

In the above code, we have added L2 regularization to the SGD optimizer with a weight decay of 0.1.

For L1 regularization, you can add it manually to the loss function:

```python
l1_lambda = 0.1
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = criterion(output, target) + l1_lambda * l1_norm
```

In the above code, we have added L1 regularization to the loss function with a lambda of 0.1.

By experimenting with these techniques, you can improve the performance of your models and prevent overfitting.