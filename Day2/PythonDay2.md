# Day 2: Basic Python Syntax and Data Types

Python is a high-level, interpreted programming language with a simple syntax that emphasizes readability and reduces the cost of program maintenance. Today, we will explore Python's syntax, standard data types, and variables.

## Task 1: Python's Syntax for Comments, Variables Declaration, and Basic Operations

Python uses the `#` symbol to start a comment. Comments are lines that exist in computer programs that are ignored by compilers and interpreters. Including comments in programs makes code more readable for humans as it provides some information or explanation about what each part of a program is doing.

```python
# This is a comment
```

In Python, variables are created when you assign a value to it. Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.

```python
x = 5
y = "Hello, World!"
```

Python supports the usual mathematical operations:

```python
# Addition
x = 5 + 2  # x will be 7

# Subtraction
x = 5 - 2  # x will be 3

# Multiplication
x = 5 * 2  # x will be 10

# Division
x = 5 / 2  # x will be 2.5
```

## Task 2: Different Data Types

In Python, every value has a datatype. Everything is an object in Python programming, and data types are classes and variables are instance (object) of these classes. Python has various standard data types that are used to define the operations possible on them and the storage method for each of them.

```python
# int
x = 5
print(type(x))  # <class 'int'>

# float
x = 5.0
print(type(x))  # <class 'float'>

# str
x = "Hello, World!"
print(type(x))  # <class 'str'>

# bool
x = True
print(type(x))  # <class 'bool'>
```

## Task 3: Type Conversion

Python defines type conversion functions to directly convert one data type to another which is useful in day to day and competitive programming.

```python
# int to float
x = 5
print(float(x))  # 5.0

# float to int
x = 5.0
print(int(x))  # 5

# int to str
x = 5
print(str(x))  # '5'

# str to int
x = "5"
print(int(x))  # 5
```

However, not all conversions are possible:

```python
# str to int
x = "Hello, World!"
print(int(x))  # ValueError: invalid literal for int() with base 10: 'Hello, World!'
```

In this case, Python is unable to convert the string "Hello, World!" to an integer, and raises a `ValueError`.