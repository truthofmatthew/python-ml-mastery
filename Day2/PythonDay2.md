# Day 2: Basic Python Syntax and Data Types

Python is a high-level, interpreted programming language with a simple syntax that emphasizes readability and reduces the cost of program maintenance. It supports multiple programming paradigms, including structured, object-oriented, and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library.

## Task 1: Python's Syntax for Comments, Variables Declaration, and Basic Operations

### Comments
In Python, comments are lines that exist in computer programs that are ignored by compilers and interpreters. Including comments in programs makes code more readable for humans as it provides some information or explanation about what each part of a program is doing. In general, it is a good idea to write comments while you are writing or updating a program as it is easy to forget your thought process later on, and comments written later may be less useful in the long term.

In Python, we use the hash (#) symbol to start writing a comment.

```python
# This is a comment
print("Hello, World!")
```

### Variables Declaration
Variables are containers for storing data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.

```python
x = 5
y = "John"
print(x)
print(y)
```

### Basic Operations
Python supports the usual mathematical operations:

```python
# Addition
print(5 + 3)  # Outputs: 8

# Subtraction
print(5 - 3)  # Outputs: 2

# Multiplication
print(5 * 3)  # Outputs: 15

# Division
print(5 / 3)  # Outputs: 1.6666666666666667

# Modulus
print(5 % 3)  # Outputs: 2

# Exponentiation
print(5 ** 3)  # Outputs: 125

# Floor division
print(5 // 3)  # Outputs: 1
```

## Task 2: Different Data Types

Python has five standard data types:

- Numbers
- String
- List
- Tuple
- Dictionary

```python
# Numbers
x = 5
print(x, type(x))

# String
x = "Hello, World!"
print(x, type(x))

# List
x = ["apple", "banana", "cherry"]
print(x, type(x))

# Tuple
x = ("apple", "banana", "cherry")
print(x, type(x))

# Dictionary
x = {"name": "John", "age": 36}
print(x, type(x))
```

## Task 3: Type Conversion

You can convert from one type to another with the `int()`, `float()`, and `str()` functions:

```python
# Convert from one type to another:

x = 5   # int
print(x, type(x))

y = str(x)
print(y, type(y))

z = float(x)
print(z, type(z))
```

Note: Type conversion can lead to data loss, and some conversions are not allowed (e.g., from a string to an integer if the string does not represent a number).