# Day 2: Python Basics - Variables, Data Types, and Operators

## Introduction

In this lesson, we will delve into the basics of Python programming. We will learn about variables, data types, and operators, which are the building blocks of any Python program. By the end of this lesson, you will be able to write simple Python scripts using these concepts.

## Learning Objectives

- Understand what variables are and how to use them in Python.
- Learn about the different data types in Python.
- Learn how to use operators to manipulate data in Python.

## Tasks

### Task 1: Understanding Variables

In Python, a variable is a named location used to store data in the memory. It is helpful to think of variables as containers that hold data. The data stored in a variable can be changed later on.

Python is a dynamically-typed language, which means you don't have to declare the data type of a variable when you create one. You can create a variable with a short name like `x`, or a more descriptive name like `carname`.

Here is an example of how to create a variable in Python:

```python
x = 5
carname = "Volvo"
```

### Task 2: Understanding Data Types

Python has several built-in data types that are commonly used in Python programs. Here are some of the most common ones:

- **Numeric Types**: `int`, `float`, `complex`
- **Sequence Types**: `list`, `tuple`, `range`
- **Text Sequence Type**: `str`
- **Mapping Type**: `dict`
- **Set Types**: `set`, `frozenset`
- **Boolean Type**: `bool`

You can use the `type()` function to find out the type of a variable. Here is an example:

```python
x = 5
print(type(x))  # Outputs: <class 'int'>
```

### Task 3: Understanding Operators

Operators are special symbols in Python that carry out arithmetic or logical computation. The value that the operator operates on is called the operand.

Here are the different types of operators in Python:

- **Arithmetic operators**: `+`, `-`, `*`, `/`, `%`, `**`, `//`
- **Comparison operators**: `==`, `!=`, `>`, `<`, `>=`, `<=`
- **Logical operators**: `and`, `or`, `not`
- **Bitwise operators**: `&`, `|`, `^`, `~`, `<<`, `>>`
- **Assignment operators**: `=`, `+=`, `-=`, `*=`, `/=`, `%=`, `//=`, `**=`, `&=`, `|=`, `^=`, `>>=`, `<<=`
- **Special operators**: `is`, `is not`, `in`, `not in`

Here is an example of how to use operators in Python:

```python
x = 10
y = 20

# Arithmetic operators
print('x + y =',x+y)  # Outputs: x + y = 30

# Comparison operators
print('x > y is',x>y)  # Outputs: x > y is False

# Logical operators
print('x < 15 and y > 15 is',x<15 and y>15)  # Outputs: x < 15 and y > 15 is True
```

In the next lesson, we will learn about control flow tools in Python, such as conditional statements and loops.