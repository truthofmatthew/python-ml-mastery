# Day 10: Regex Compilation and Efficiency

Regular expressions (regex) are a powerful tool for pattern matching and data extraction in Python. However, they can be computationally expensive, especially when dealing with large datasets or complex patterns. In this tutorial, we will explore how to compile regex patterns for efficiency and how to write efficient regex patterns.

## Task 1: Compile a Complex Regex Pattern

The `re` module in Python provides the `compile()` function, which compiles a regex pattern into a regex object. This object can be used for matching and searching, and it's more efficient when the pattern is used multiple times.

Let's compile a complex regex pattern that matches email addresses:

```python
import re

# Define the regex pattern
pattern = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"

# Compile the pattern
compiled_pattern = re.compile(pattern)
```

Now, `compiled_pattern` is a regex object that can be used for matching:

```python
# Use the compiled pattern
match = compiled_pattern.match("example@example.com")
```

## Task 2: Measure Performance Difference Between Compiled and Non-Compiled Regex

To measure the performance difference, we can use the `timeit` module in Python. Let's compare the time it takes to match a string using a compiled pattern and a non-compiled pattern:

```python
import timeit

# Time the non-compiled pattern
start_time = timeit.default_timer()
re.match(pattern, "example@example.com")
end_time = timeit.default_timer()
non_compiled_time = end_time - start_time

# Time the compiled pattern
start_time = timeit.default_timer()
compiled_pattern.match("example@example.com")
end_time = timeit.default_timer()
compiled_time = end_time - start_time

print(f"Non-compiled time: {non_compiled_time}")
print(f"Compiled time: {compiled_time}")
```

You should see that the compiled pattern is faster, especially if the pattern is used multiple times.

## Task 3: Rewrite an Inefficient Regex Pattern to Be More Efficient

Efficient regex writing involves minimizing the use of certain costly operations, such as backtracking and greedy quantifiers. Let's take an inefficient pattern that matches any string of characters between two quotes:

```python
inefficient_pattern = r"\".*\""
```

This pattern uses the `.*` quantifier, which is greedy and causes a lot of backtracking. A more efficient version would use the non-greedy quantifier `.*?`:

```python
efficient_pattern = r"\".*?\""
```

This pattern will match the shortest possible string between two quotes, reducing the amount of backtracking.

In conclusion, compiling regex patterns and writing efficient patterns can significantly improve the performance of your Python code. Always consider these techniques when working with regular expressions.