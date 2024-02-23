# Day 10: Working with External Libraries and pip

Python's rich ecosystem of third-party libraries is one of its most powerful features. These libraries, also known as packages, extend Python's functionality, allowing you to perform a wide range of tasks without having to write the code from scratch. In this tutorial, we will explore how to use pip, Python's package manager, to install and manage these packages and their dependencies.

## Task 1: Use pip to install a popular third-party library like requests or numpy

pip is a command-line tool that allows you to install and manage Python packages. To install a package, you simply need to run the command `pip install <package-name>`. For example, to install the requests library, you would run:

```bash
pip install requests
```

And to install numpy, you would run:

```bash
pip install numpy
```

You can also use pip to upgrade a package to the latest version with the `--upgrade` flag:

```bash
pip install --upgrade requests
```

## Task 2: Write a Python script that uses the requests library to download the content of a webpage

The requests library is a popular Python package for making HTTP requests. It abstracts the complexities of making requests behind a beautiful, simple API, allowing you to send HTTP/1.1 requests with various methods like GET, POST, and others. With it, you can work with REST services, like downloading the content of a webpage.

Here is a simple script that uses requests to download and print the content of a webpage:

```python
import requests

response = requests.get('https://www.example.com')

# Check if the request was successful
if response.status_code == 200:
    print(response.text)
else:
    print('Failed to download webpage:', response.status_code)
```

## Task 3: Explore the functionalities of numpy by creating and manipulating a multidimensional array

NumPy is a powerful library for numerical computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.

Here is an example of creating a 2D array in NumPy and performing some operations on it:

```python
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print('Original array:')
print(arr)

# Multiply all elements of the array by 2
arr2 = arr * 2

print('Array after multiplication:')
print(arr2)

# Calculate the sum of all elements in the array
sum = np.sum(arr)

print('Sum of all elements:', sum)
```

In this tutorial, we have explored how to use pip to install and manage Python packages, and how to use two popular packages, requests and NumPy. Python's rich ecosystem of third-party libraries is one of its greatest strengths, and learning how to use these libraries effectively is an important skill for any Python programmer.