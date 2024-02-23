# Day 4: Functions in Python

Functions in Python are blocks of reusable code that perform a specific task. They help to organize, modularize, and increase the reusability of your code. In this tutorial, we will learn how to define and use functions, understand the scope of variables, and work with different types of function arguments.

## Task 1: Define a Function

Let's start by defining a simple function that takes two numbers as arguments and returns their sum.

```python
def add_numbers(num1, num2):
    return num1 + num2

# Call the function
result = add_numbers(5, 10)
print(result)  # Output: 15
```

In the above code, `add_numbers` is a function that takes two arguments, `num1` and `num2`. The `return` statement is used to exit a function and return a value.

## Task 2: Default Argument Values

Python allows function arguments to have default values. If the function is called without the argument, it uses the default value.

Let's create a function that greets a user. If no name is provided, it defaults to "World".

```python
def greet(name="World"):
    return "Hello, " + name + "!"

# Call the function
print(greet())  # Output: Hello, World!
print(greet("Alice"))  # Output: Hello, Alice!
```

In the above code, `name` is an argument with a default value of "World". If we call `greet()` without an argument, it uses the default value.

## Task 3: Positional and Keyword Arguments

In Python, there are two types of arguments that can be used when calling a function - positional and keyword arguments.

Positional arguments are arguments that need to be in the correct positional order and number to match the parameters in the function definition.

Keyword arguments are arguments identified by the parameter name. This allows you to skip arguments or place them out of order because Python's interpreter can use the provided keywords to match the values with parameters.

Let's create a function that requires three arguments, then call it in different ways to understand the difference between positional and keyword arguments.

```python
def display_info(name, age, country):
    print("Name:", name)
    print("Age:", age)
    print("Country:", country)

# Positional arguments
display_info("Alice", 25, "USA")

# Keyword arguments
display_info(age=30, country="UK", name="Bob")

# Mix of positional and keyword arguments
display_info("Charlie", country="Canada", age=35)
```

In the above code, the `display_info` function requires three arguments. When we call this function, we can use positional arguments, keyword arguments, or a mix of both.

Remember, when using a mix of positional and keyword arguments, all positional arguments must come before keyword arguments.

That's it for today's lesson on functions in Python. Practice what you've learned and stay tuned for more Python tutorials.