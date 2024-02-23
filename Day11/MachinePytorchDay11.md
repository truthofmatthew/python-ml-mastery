# Day 11: Understanding Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or the spoken word. Unlike traditional neural networks, RNNs have loops, allowing information to persist.

## Task 1: Define an RNN architecture using PyTorch modules

To define an RNN architecture in PyTorch, we use the `nn.RNN` module. Here is a simple example:

```python
import torch
from torch import nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device) 
        out, _ = self.rnn(x, h0)  
        out = self.fc(out[:, -1, :])  
        return out
```

In this code, `input_size` is the number of features in the input, `hidden_size` is the number of features in the hidden state, and `output_size` is the number of features in the output.

## Task 2: Prepare a time series or text dataset for sequence modeling

Preparing a dataset for sequence modeling involves transforming the data into a suitable format for the RNN. For a time series dataset, this could involve creating sequences of a fixed length from the time series data. For a text dataset, this could involve tokenizing the text and creating sequences of a fixed length from the tokens.

Here is an example of how to prepare a time series dataset:

```python
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(dataset, seq_length)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
```

In this code, `create_sequences` function creates sequences of length `seq_length` from the dataset. Each sequence in `X` is associated with a target value in `y`, which is the value that comes after the sequence in the dataset.

## Task 3: Train your RNN model and evaluate its performance on a test set

Training an RNN model involves feeding the sequences and their associated target values into the model, calculating the loss, and updating the model's weights. Here is an example:

```python
model = SimpleRNN(input_size=1, hidden_size=32, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print('Epoch:', epoch+1, 'Loss:', loss.item())
```

In this code, we first create an instance of our `SimpleRNN` model. We then define a loss function (`nn.MSELoss` for a regression task) and an optimizer (`torch.optim.Adam`). We then train the model for 100 epochs, updating the model's weights in each epoch to minimize the loss.

Evaluating the model's performance on a test set involves feeding the test sequences into the model and comparing the model's predictions with the actual values. Here is an example:

```python
model.eval()
test_output = model(test_X)
test_loss = criterion(test_output, test_y)
print('Test Loss:', test_loss.item())
```

In this code, we first switch the model to evaluation mode with `model.eval()`. We then feed the test sequences (`test_X`) into the model and calculate the loss with the actual values (`test_y`).