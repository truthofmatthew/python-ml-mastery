# Day 11: Regular Expressions in Python

Regular expressions, also known as regex, are a powerful tool for working with text. They allow you to create sophisticated patterns to match, locate, and manage text strings. Python's `re` module provides support for regular expressions, and we will explore its functionalities in this tutorial.

## Task 1: Write a Regular Expression to Validate Email Addresses

To validate an email address, we need to ensure it follows the general pattern of `username@domain.extension`. Here's a simple regular expression that matches this pattern:

```python
import re

def validate_email(email):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.match(pattern, email):
        return True
    return False
```

This pattern checks for one or more alphanumeric characters (including ., _, %, +, and -) followed by @, followed by one or more alphanumeric characters (including . and -), followed by a period, and finally, two or more alphabetic characters.

## Task 2: Use Regular Expressions to Find All Dates in a Given Text

To find all dates in a text, we can use a regular expression that matches common date formats. Here's an example that matches dates in the format `dd-mm-yyyy` or `dd/mm/yyyy`:

```python
def find_dates(text):
    pattern = r'\b(0?[1-9]|[12][0-9]|3[01])[-/](0?[1-9]|1[012])[-/](19|20)\d\d\b'
    return re.findall(pattern, text)
```

This pattern matches one or two digits (01 to 31) for the day, followed by a dash or slash, followed by one or two digits (01 to 12) for the month, another dash or slash, and finally, a four-digit year starting with 19 or 20.

## Task 3: Replace All Occurrences of a Pattern in a String with Another String

The `re` module's `sub()` function allows you to replace all occurrences of a pattern in a string with another string. Here's an example:

```python
def replace_pattern(text, pattern, replacement):
    return re.sub(pattern, replacement, text)
```

For example, to replace all occurrences of 'python' with 'Python' in a string, you would call `replace_pattern(text, 'python', 'Python')`.

Remember, regular expressions are a powerful tool but can be complex. Always test your patterns to ensure they match what you expect.