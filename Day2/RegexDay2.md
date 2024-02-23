# Day 2: Character Sets and Negation in Regex

## Introduction

In the previous lesson, we learned about the basics of regex in Python, including simple matching and basic syntax. Today, we will delve deeper into regex by exploring character sets and their negations.

Character sets allow us to match any character from a specific set of characters. They are defined by enclosing a set of characters inside square brackets `[]`. For example, the regex `[abc]` will match any of the characters `a`, `b`, or `c`.

Negation in regex is used to match any character that is not in a specific set. It is defined by placing a caret `^` at the start of the character set. For example, the regex `[^abc]` will match any character that is not `a`, `b`, or `c`.

## Task 1: Create a regex to match any vowel

To match any vowel, we can use a character set that includes all the vowels. In regex, this can be written as `[aeiou]`.

```python
import re

text = "Hello, world!"
matches = re.findall('[aeiou]', text, re.IGNORECASE)
print(matches)  # Output: ['e', 'o', 'o']
```

In this example, `re.findall('[aeiou]', text, re.IGNORECASE)` returns all the vowels in the string `text`. The `re.IGNORECASE` flag makes the regex case-insensitive, so it matches both lowercase and uppercase vowels.

## Task 2: Write a regex to find non-digit characters

To find non-digit characters, we can use a negated character set that includes all digits. In regex, this can be written as `[^0-9]`.

```python
import re

text = "123abc456def"
matches = re.findall('[^0-9]', text)
print(matches)  # Output: ['a', 'b', 'c', 'd', 'e', 'f']
```

In this example, `re.findall('[^0-9]', text)` returns all the non-digit characters in the string `text`.

## Task 3: Use a regex character set to match specific letters in a string

To match specific letters in a string, we can use a character set that includes those letters. For example, to match the letters `a`, `b`, and `c`, we can use the regex `[abc]`.

```python
import re

text = "abcdefg"
matches = re.findall('[abc]', text)
print(matches)  # Output: ['a', 'b', 'c']
```

In this example, `re.findall('[abc]', text)` returns all occurrences of the letters `a`, `b`, and `c` in the string `text`.

In conclusion, character sets and their negations are powerful tools in regex that allow us to match specific sets of characters. They are especially useful when we want to match any character from a specific set, or any character that is not in a specific set.