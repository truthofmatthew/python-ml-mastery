# Day 8: Data Loading and Processing with PyTorch

In this tutorial, we will learn how to use PyTorch's `DataLoader` and `Dataset` classes for efficient data handling. These classes are essential for training machine learning models, as they allow us to easily iterate over our data in batches and apply transformations to preprocess our data.

## Task 1: Create a Custom Dataset Class

The first step in handling data with PyTorch is to create a custom `Dataset` class. This class should inherit from PyTorch's `Dataset` class and override the `__len__` and `__getitem__` methods.

Here is an example of a custom `Dataset` class for a simple dataset of images and labels:

```python
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

In this class, `image_paths` is a list of paths to the image files, `labels` is a list of labels for the images, and `transform` is an optional argument for image transformations.

## Task 2: Use DataLoader to Iterate Over Batches of Data

Once we have our `Dataset` class, we can use PyTorch's `DataLoader` class to easily iterate over our data in batches. Here is an example:

```python
from torch.utils.data import DataLoader

dataset = MyDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    # Training code here...
```

In this example, `DataLoader` takes our dataset, a batch size, and a boolean indicating whether to shuffle the data. It returns an iterator that yields batches of images and labels.

## Task 3: Apply Transformations to Preprocess the Data

Finally, we can use PyTorch's `transforms` module to apply transformations to our data. These transformations can include resizing, normalization, data augmentation, and more.

Here is an example of how to apply transformations to our images:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MyDataset(image_paths, labels, transform)
```

In this example, we first resize the images to 224x224 pixels, then convert them to PyTorch tensors, and finally normalize them with the mean and standard deviation of the ImageNet dataset.

That's it for today's tutorial. With these tools, you should be able to handle data efficiently in PyTorch.