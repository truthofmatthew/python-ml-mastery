# Day 2: Basic Python Syntax and Data Types

Python is a high-level, interpreted programming language with a simple syntax that emphasizes readability and reduces the cost of program maintenance. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. 

## Task 1: Python's Syntax for Comments, Variables Declaration, and Basic Operations

### Comments
In Python, comments are lines that exist in computer programs that are ignored by compilers and interpreters. Including comments in programs makes code more readable for humans as it provides some information or explanation about what each part of a program is doing. In general, it is a good practice to comment your code.

In Python, we use the hash (`#`) symbol to start writing a comment.

```python
# This is a comment
print("Hello, World!")
```

### Variables Declaration
In Python, variables are created the moment you first assign a value to it. You do not have to declare the data type of the variable. This is handled internally according to the type of value you assign to the variable.

```python
x = 5          # x is of type int
y = "Python"   # y is now of type str
```

### Basic Operations
Python supports the usual mathematical operations:

```python
x = 10
y = 5

print(x + y)  # Addition
print(x - y)  # Subtraction
print(x * y)  # Multiplication
print(x / y)  # Division
print(x % y)  # Modulus
print(x ** y) # Exponentiation
print(x // y) # Floor division
```

## Task 2: Different Data Types

Python has five standard data types:

- Numbers
- String
- List
- Tuple
- Dictionary

Let's create a script that demonstrates the use of different data types and prints the type of each variable.

```python
x = 5          # int
y = 5.5        # float
z = "Python"   # str
a = True       # bool

print(type(x))
print(type(y))
print(type(z))
print(type(a))
```

## Task 3: Type Conversion

Python defines type conversion functions to directly convert one data type to another.

```python
x = 5          # int
y = 5.5        # float
z = "Python"   # str

# convert from int to float:
a = float(x)

# convert from float to int:
b = int(y)

# convert from int to string:
c = str(x)

print(a)
print(b)
print(c)
```

Note: Type conversion can lead to data loss, as in the conversion from float to int, the digits after the decimal point are discarded. Also, not all types of conversion are possible. For example, you cannot convert a string that does not represent a number into an integer or a float. Trying to do so will raise a `ValueError`.