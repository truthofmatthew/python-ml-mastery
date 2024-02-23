# Day 6: Error Handling in Python

## Brief Introduction

In the process of writing and executing Python programs, you may encounter various types of errors. These errors, also known as exceptions, can disrupt the normal flow of your program. However, Python provides mechanisms to handle these exceptions gracefully, allowing your program to continue running or terminate in a controlled manner. In this tutorial, we will explore how to predict, catch, and handle exceptions in Python.

## Learning Objective

By the end of this tutorial, you will be able to understand different types of exceptions and implement robust error handling in your Python programs.

## Task 1: Force a Division by Zero Error and Catch It with a Try-Except Block

In Python, dividing a number by zero raises a `ZeroDivisionError`. Let's force this error and handle it using a `try-except` block.

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
```

In the above code, the `try` block contains the code that may raise an exception. If an exception occurs, the `except` block is executed. Here, we catch the `ZeroDivisionError` and print a custom error message.

## Task 2: Use the Try-Except-Else-Finally Structure to Handle File Reading and Closing Safely

When working with files, it's important to ensure that the file is properly closed after use, even if an error occurs. The `try-except-else-finally` structure can be used for this purpose.

```python
try:
    file = open('file.txt', 'r')
    content = file.read()
except FileNotFoundError:
    print("Error: The file does not exist.")
else:
    print("File read successfully.")
finally:
    file.close()
    print("File closed.")
```

In this code, the `try` block attempts to open and read a file. If the file does not exist, a `FileNotFoundError` is raised and caught in the `except` block. If no exception is raised, the `else` block is executed. Regardless of whether an exception occurs, the `finally` block is always executed, ensuring that the file is closed.

## Task 3: Raise a Custom Exception in a Situation Where Input Data Does Not Meet a Certain Condition

Sometimes, you may want to raise an exception when a specific condition is not met. This can be done using the `raise` statement.

```python
def check_age(age):
    if age < 18:
        raise ValueError("Error: Age must be at least 18.")
    else:
        print("Age is valid.")

try:
    check_age(15)
except ValueError as e:
    print(e)
```

In this code, the `check_age` function raises a `ValueError` if the input age is less than 18. The `try-except` block catches this exception and prints the error message.

## Conclusion

Error handling is a crucial aspect of programming in Python. It allows you to predict and handle potential errors, making your programs more robust and reliable. By understanding and implementing the concepts covered in this tutorial, you can write Python programs that gracefully handle exceptions.