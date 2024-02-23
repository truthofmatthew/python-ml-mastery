# Day 2: Python Basics - Variables, Data Types, and Operators

## Brief Introduction

Python, like any other programming language, uses variables to store data. These variables can hold different types of data, such as integers, floating-point numbers, strings, and more. Python also provides a set of operators that allow you to perform operations on these variables.

## Learning Objective

By the end of this lesson, you should be able to understand and use Python variables, data types, and operators.

## Tasks

### Task 1: Understanding Variables

In Python, a variable is a named location used to store data in memory. It is helpful to think of variables as containers that hold data. The data stored in a variable can be changed during program execution.

```python
# Example of Python Variables
x = 10
y = "Hello, World!"
```

In the above example, `x` and `y` are variables. `x` stores an integer value `10`, and `y` stores a string value `"Hello, World!"`.

### Task 2: Understanding Data Types

Python has several built-in data types. Some of the most commonly used data types are:

- **Integers**: They are used to store numeric data. Example: `x = 10`
- **Float**: They are used to store decimal numbers. Example: `y = 20.5`
- **Strings**: They are used to store text data. Example: `z = "Hello, World!"`
- **Booleans**: They are used to store True or False values. Example: `is_valid = True`

Python is a dynamically typed language, which means you don't have to declare the data type of a variable when you create it.

### Task 3: Understanding Operators

Python provides several types of operators, which are symbols that carry out arithmetic or logical computation. The value that the operator operates on is called the operand.

- **Arithmetic operators**: These operators are used to perform mathematical operations like addition, subtraction, multiplication, etc. Example: `+, -, *, /, %, **, //`
- **Comparison operators**: These operators are used to compare values. It either returns True or False according to the condition. Example: `==, !=, <, >, <=, >=`
- **Logical operators**: These operators are used to combine conditional statements. Example: `and, or, not`

```python
# Example of Python Operators
x = 10
y = 20

# Arithmetic Operators
print("x + y = ", x + y)  # Output: x + y = 30

# Comparison Operators
print("x > y = ", x > y)  # Output: x > y = False

# Logical Operators
print("x < 15 and y > 15 = ", x < 15 and y > 15)  # Output: x < 15 and y > 15 = True
```

In the above example, `+` is an arithmetic operator, `>` is a comparison operator, and `and` is a logical operator.