# Python Data Structures - Lists and Dictionaries

Python provides several built-in data structures that make it easy to store and manipulate data. In this tutorial, we will focus on two of the most commonly used data structures: lists and dictionaries.

## Lists

A list in Python is an ordered collection of items. Lists are mutable, which means that you can add, remove, or change items after the list is created.

### Creating a List

You can create a list by placing a comma-separated sequence of items inside square brackets `[]`.

```python
# Create a list of integers
numbers = [1, 2, 3, 4, 5]
print(numbers)  # Output: [1, 2, 3, 4, 5]
```

### Adding and Removing Elements

You can add an element to the end of the list using the `append()` method, and remove an element with the `remove()` method.

```python
# Add an element
numbers.append(6)
print(numbers)  # Output: [1, 2, 3, 4, 5, 6]

# Remove an element
numbers.remove(1)
print(numbers)  # Output: [2, 3, 4, 5, 6]
```

### Sorting a List

You can sort a list in ascending order with the `sort()` method.

```python
# Sort the list
numbers.sort()
print(numbers)  # Output: [2, 3, 4, 5, 6]
```

## List Comprehensions

List comprehensions provide a concise way to create lists based on existing lists. Here is how you can create a list of square numbers from 1 to 10.

```python
# List comprehension to generate a list of square numbers
squares = [number**2 for number in range(1, 11)]
print(squares)  # Output: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

## Dictionaries

A dictionary in Python is an unordered collection of items. Each item in a dictionary has a key and a value. You can use the key to access the corresponding value.

### Creating a Dictionary

You can create a dictionary by placing a comma-separated list of key-value pairs inside curly braces `{}`. The key and value are separated by a colon `:`.

```python
# Create a dictionary mapping strings to their lengths
string_lengths = {"Python": 6, "data": 4, "structures": 10}
print(string_lengths)  # Output: {'Python': 6, 'data': 4, 'structures': 10}
```

### Accessing and Modifying Data

You can access the value of a specific key using square brackets `[]`.

```python
# Access the length of "Python"
print(string_lengths["Python"])  # Output: 6
```

You can also modify the value of a specific key.

```python
# Change the length of "Python" to 7
string_lengths["Python"] = 7
print(string_lengths)  # Output: {'Python': 7, 'data': 4, 'structures': 10}
```

In this tutorial, you learned how to create and manipulate lists and dictionaries in Python. These data structures are powerful tools that you can use to store and organize your data.