# Day 9: Working with Convolutional Neural Networks (CNNs)

Convolutional Neural Networks (CNNs) are a class of deep learning models, primarily used for image recognition and processing tasks. They are designed to automatically and adaptively learn spatial hierarchies of features from tasks with grid-like topology, such as images.

In this tutorial, we will build and train a basic CNN for image classification. We will define the architecture of the CNN, understand and apply pooling and convolutional layers, and train the CNN on an image dataset to evaluate its performance.

## Task 1: Define a CNN Architecture for Image Classification

A typical CNN architecture consists of Convolutional layers, Pooling layers, and Fully Connected layers. Let's define a simple CNN architecture using PyTorch.

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)  # Convolutional layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer
        self.fc3 = nn.Linear(84, 10)  # Fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU activation function after conv1 and then max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply ReLU activation function after conv2 and then max pooling
        x = x.view(-1, 16 * 5 * 5)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Apply ReLU activation function after fc1
        x = F.relu(self.fc2(x))  # Apply ReLU activation function after fc2
        x = self.fc3(x)  # No activation function after the final layer
        return x

net = Net()
```

## Task 2: Understand and Apply Pooling and Convolutional Layers

Convolutional layers apply a convolution operation on the input layer to produce a set of output layers, often referred to as feature maps. Pooling layers reduce the spatial size of the convolved feature, which decreases the computational complexity for the network.

In the above code, `nn.Conv2d(3, 6, 5)` creates a convolutional layer that takes an input with 3 channels, applies 6 filters, each of size 5x5. `nn.MaxPool2d(2, 2)` applies a 2x2 max pooling over the input, reducing the height and width by a factor of 2.

## Task 3: Train the CNN on an Image Dataset and Evaluate Performance

To train the CNN, we need a loss function and an optimizer. We'll use Cross-Entropy loss and Stochastic Gradient Descent (SGD) with momentum.

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Now, let's train the network. We'll run two epochs for simplicity.

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

To evaluate the performance of the network, we can check the accuracy on the test dataset.

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
```

This concludes our tutorial on working with Convolutional Neural Networks. We have learned how to define a CNN architecture, understand and apply pooling and convolutional layers, and train and evaluate a CNN on an image dataset.