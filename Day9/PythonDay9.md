# Day 9: Iterators and Generators in Python

Python provides several tools to make data processing more efficient. Among these are iterators and generators, which allow for advanced iteration techniques and efficient looping. This tutorial will guide you through creating custom iterators and generator functions, as well as using generator expressions for lazy evaluation.

## Task 1: Create a Custom Iterator

An iterator in Python is an object that can be iterated (looped) upon. An object which will return data, one element at a time. Let's create a custom iterator that yields squares of numbers up to a certain limit.

```python
class SquareIterator:
    def __init__(self, limit):
        self.limit = limit
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < self.limit:
            result = self.n ** 2
            self.n += 1
            return result
        else:
            raise StopIteration

# Usage
squares = SquareIterator(5)
for square in squares:
    print(square)
```

In this example, `SquareIterator` is an iterable object that yields the squares of numbers from 0 up to (but not including) the limit. The `__iter__` method returns the iterator object itself, and the `__next__` method returns the next value from the iterator.

## Task 2: Write a Generator Function

A generator in Python is a function that behaves like an iterator. It allows you to write iterators much like the one above but in a much shorter and cleaner syntax. Let's write a generator function that generates an infinite sequence of Fibonacci numbers.

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Usage
fib = fibonacci()
for i in range(10):
    print(next(fib))
```

In this example, `fibonacci` is a generator function that yields the Fibonacci sequence indefinitely. The `yield` keyword is used to define a generator, replacing the `return` of a function to provide a result to its caller without destroying local variables.

## Task 3: Use a Generator Expression

A generator expression is a high performance, memory efficient generalization of list comprehensions and generators. Let's use a generator expression to lazily evaluate and yield results from a large dataset.

```python
# Assume we have a large list of numbers
numbers = range(1000000)

# We want to calculate the square of all numbers, but we don't want to store the entire result in memory
squares = (n**2 for n in numbers)

# Usage
for i in range(10):
    print(next(squares))
```

In this example, `squares` is a generator expression that lazily yields the squares of numbers from the `numbers` list. This is memory efficient because it doesn't calculate all the squares at once; instead, it calculates each square on the fly as you loop through the generator.