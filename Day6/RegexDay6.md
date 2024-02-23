# Day 6: Flags and Modifiers in Regular Expressions

Regular expressions (regex) are a powerful tool for pattern matching and data extraction. However, their behavior can be modified using flags to suit specific needs. In Python's `re` module, these flags are passed as an optional argument to various functions. Today, we will explore some of these flags and their applications.

## Task 1: Use case-insensitive flag in a regex match

The `re.IGNORECASE` or `re.I` flag allows for case-insensitive matching. This means that the regex pattern will match regardless of case.

Consider the following example:

```python
import re

text = "Hello World"
pattern = r"hello"

# Without the re.I flag
print(re.search(pattern, text))  # Returns: None

# With the re.I flag
print(re.search(pattern, text, re.I))  # Returns: <re.Match object; span=(0, 5), match='Hello'>
```

In the first `re.search()`, the pattern "hello" does not match "Hello" because of the case difference. However, when we use the `re.I` flag in the second `re.search()`, it matches "Hello" regardless of the case difference.

## Task 2: Match across multiple lines with a flag

The `re.MULTILINE` or `re.M` flag allows start and end metacharacters (`^` and `$`) to match at the beginning and end of each line (defined by `\n`) rather than just the start and end of the string.

Consider the following example:

```python
import re

text = """Hello World
Hello Universe"""

pattern = r"^Hello"

# Without the re.M flag
print(re.findall(pattern, text))  # Returns: ['Hello']

# With the re.M flag
print(re.findall(pattern, text, re.M))  # Returns: ['Hello', 'Hello']
```

In the first `re.findall()`, the pattern "^Hello" matches only the first "Hello" at the start of the string. However, when we use the `re.M` flag in the second `re.findall()`, it matches both "Hello" at the start of each line.

## Task 3: Create a regex pattern using the verbose flag to improve readability

The `re.VERBOSE` or `re.X` flag allows you to write more readable regular expressions. It ignores whitespace (except inside a set `[]` or when escaped by `\`) and treats unescaped `#` as a comment marker.

Consider the following example:

```python
import re

text = "Hello World"

# Without the re.X flag
pattern = r"^H\w*o"
print(re.search(pattern, text))  # Returns: <re.Match object; span=(0, 5), match='Hello'>

# With the re.X flag
pattern = r"""
^    # Start of the string
H    # Literal character 'H'
\w*  # Any word character (a-z, A-Z, 0-9, _), zero or more times
o    # Literal character 'o'
"""
print(re.search(pattern, text, re.X))  # Returns: <re.Match object; span=(0, 5), match='Hello'>
```

In the first `re.search()`, the pattern is a bit hard to understand. However, when we use the `re.X` flag in the second `re.search()`, we can write the pattern in a more readable way with comments.

In conclusion, flags can greatly enhance the flexibility and readability of regular expressions. They are a powerful tool in your regex arsenal.