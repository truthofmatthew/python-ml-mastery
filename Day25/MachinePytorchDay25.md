# Day 25: Creating Custom Layers and Models in PyTorch

PyTorch provides a wide range of pre-defined layers and models for deep learning. However, there are situations where you may need to create your own custom layers or models to suit specific requirements. This tutorial will guide you through the process of creating custom layers and models in PyTorch.

## Task 1: Understand how to extend nn.Module to create custom layers

PyTorch allows us to create custom layers by extending the `nn.Module` class. This class provides the base functionality for all neural network modules, which includes layers and models. 

To create a custom layer, we need to define a new class that inherits from `nn.Module`. This class should implement two methods: `__init__()` and `forward()`. 

The `__init__()` method is used to initialize the layer and define its parameters. The `forward()` method defines the computation performed at every call. 

Here is an example of a custom layer that applies a linear transformation to its input:

```python
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomLinear, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        return torch.mm(x, self.weights) + self.bias
```

## Task 2: Implement a custom layer/model for a specific task

Let's implement a custom model for a simple task. We will create a multi-layer perceptron (MLP) with two hidden layers. 

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = CustomLinear(input_dim, hidden_dim)
        self.layer2 = CustomLinear(hidden_dim, hidden_dim)
        self.layer3 = CustomLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
```

In this model, we use our `CustomLinear` layer and the ReLU activation function. The `forward()` method defines the flow of data through the model.

## Task 3: Integrate your custom layer/model in a training loop and evaluate

Once we have defined our custom model, we can use it in a training loop just like any other PyTorch model. Here is an example:

```python
# Initialize the model, loss function, and optimizer
model = MLP(input_dim=784, hidden_dim=256, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

In this training loop, we first initialize our custom model, a loss function, and an optimizer. Then, for each epoch and each batch of data, we perform a forward pass through the model, compute the loss, and update the model's parameters.

To evaluate the model, we can use it to make predictions and compare them to the true labels:

```python
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
```

In this code, we set the model to evaluation mode with `model.eval()`, then iterate over the test data, making predictions with the model and comparing them to the true labels to compute the accuracy.