# Day 12: Decorators in Python

Decorators in Python are a powerful tool that allows us to wrap a function or method in order to extend or modify its behavior dynamically, without permanently modifying it. This is achieved by defining a wrapper function that modifies the behavior of the function it decorates.

## Task 1: Write a decorator that times the execution of a function

Let's start by creating a simple decorator that measures the time it takes to execute a function. We'll use Python's built-in `time` module for this.

```python
import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper
```

To use this decorator, simply prepend it to the function definition using the `@` symbol:

```python
@timer_decorator
def long_running_function():
    time.sleep(2)
```

## Task 2: Create a decorator that prints the arguments passed to any function

Next, let's create a decorator that prints the arguments passed to a function. This can be useful for debugging purposes.

```python
def print_args_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments: {args}, Keyword Arguments: {kwargs}")
        return func(*args, **kwargs)
    return wrapper
```

Again, to use this decorator, prepend it to the function definition:

```python
@print_args_decorator
def function_with_args(a, b, c=1, d=2):
    pass
```

## Task 3: Implement a memoization decorator to optimize recursive function calls

Finally, let's implement a memoization decorator. Memoization is a technique used to optimize recursive function calls by storing the results of expensive function calls and reusing them when the same inputs occur again.

```python
def memoize_decorator(func):
    cache = dict()

    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            result = func(*args)
            cache[args] = result
            return result
    return wrapper
```

This decorator can be used to optimize recursive functions, such as the Fibonacci sequence:

```python
@memoize_decorator
def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
```

With these examples, you should now have a good understanding of how to create and apply decorators in Python. Remember, decorators are a powerful tool that can greatly enhance your code when used correctly.