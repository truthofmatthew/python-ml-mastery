# Day 4: Building Your First Neural Network in PyTorch

In this tutorial, we will learn how to build a simple neural network using PyTorch. We will define a neural network architecture, implement the forward pass, and experiment with different activation functions.

## Task 1: Define a Neural Network Architecture Using nn.Module

PyTorch provides a base class `nn.Module` that we can use to define our neural network architecture. Let's create a simple feed-forward neural network with one hidden layer.

```python
import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        pass  # We will fill this in Task 2
```

In the above code, `nn.Linear` is a linear layer with input and output dimensions specified. The `forward` function is where we define how our input data flows through our network.

## Task 2: Implement the Forward Pass for Your Network

The forward pass is the process of transforming the input data into output. In our case, the input data will pass through two linear layers. Let's implement this in the `forward` function.

```python
def forward(self, x):
    out = self.fc1(x)
    out = self.fc2(out)
    return out
```

Now, our input data `x` will first pass through the first linear layer `fc1`, and then the output of `fc1` will pass through the second linear layer `fc2`.

## Task 3: Experiment with Different Activation Functions

Activation functions introduce non-linearity into our model, allowing it to learn more complex patterns. PyTorch provides several activation functions in the `torch.nn.functional` module. Let's experiment with the ReLU (Rectified Linear Unit) activation function.

```python
import torch.nn.functional as F

def forward(self, x):
    out = self.fc1(x)
    out = F.relu(out)  # Apply ReLU after fc1
    out = self.fc2(out)
    return out
```

In the above code, we applied the ReLU activation function to the output of `fc1` before passing it to `fc2`. This allows our model to learn non-linear patterns in the data.

In this tutorial, we learned how to define a neural network architecture, implement the forward pass, and use activation functions in PyTorch. In the next tutorial, we will learn how to train our neural network.