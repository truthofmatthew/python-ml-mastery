# Day 1: Basic Syntax and Simple Matching

## Introduction

Regular expressions, often abbreviated as regex, are a powerful tool for manipulating text. They are a sequence of characters that form a search pattern. This pattern can be used to match, locate, and manage text. Python's `re` module provides support for regular expressions.

## Learning Objective

By the end of this lesson, you will be able to understand the basics of regex in Python and use it to match digits, find whitespace characters, and split strings by punctuation.

## Task 1: Write a regex to match any digit

In regex, `\d` is used to match any digit (0-9). Let's see how to use it in Python.

```python
import re

# Define the pattern
pattern = r'\d'

# Define the string
string = 'The year is 2021.'

# Use re.findall() to find all matches
matches = re.findall(pattern, string)

print(matches)  # Output: ['2', '0', '2', '1']
```

In the above code, `r'\d'` is the regex pattern that matches any digit. `re.findall()` is used to find all matches in the string.

## Task 2: Create a regex to find all whitespace characters in a string

In regex, `\s` is used to match any whitespace character (spaces, tabs, line breaks). Let's see how to use it in Python.

```python
import re

# Define the pattern
pattern = r'\s'

# Define the string
string = 'Hello, World! How are you?'

# Use re.findall() to find all matches
matches = re.findall(pattern, string)

print(matches)  # Output: [' ', ' ', ' ', ' ', ' ']
```

In the above code, `r'\s'` is the regex pattern that matches any whitespace character. `re.findall()` is used to find all matches in the string.

## Task 3: Use regex to split a string by any punctuation

In regex, `\W` is used to match any non-word character, which includes punctuation. Let's see how to use it in Python.

```python
import re

# Define the pattern
pattern = r'\W'

# Define the string
string = 'Hello, World! How are you?'

# Use re.split() to split the string
split_string = re.split(pattern, string)

print(split_string)  # Output: ['Hello', '', 'World', '', 'How', 'are', 'you', '']
```

In the above code, `r'\W'` is the regex pattern that matches any non-word character. `re.split()` is used to split the string by the matches.

In the next lesson, we will learn more about regex patterns and how to use them to match more complex patterns in strings.