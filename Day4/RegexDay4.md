# Day 4: Grouping and Capturing in Regex

Grouping and capturing are powerful features in regular expressions (regex) that allow you to match multiple patterns and extract the matched strings. This tutorial will guide you through the process of grouping multiple patterns, using regex groups to find and replace a pattern in a string, and creating a regex to capture and extract phone numbers from text.

## Task 1: Write a regex to group multiple patterns

In regex, parentheses `()` are used to group multiple patterns together. This allows you to apply quantifiers to the entire group or to isolate a part of the regex.

Consider the following example where we want to match the strings "abc", "def", or "ghi":

```python
import re

pattern = r"(abc|def|ghi)"
text = "abc def ghi jkl"
matches = re.findall(pattern, text)

print(matches)  # Output: ['abc', 'def', 'ghi']
```

In the pattern `r"(abc|def|ghi)"`, the parentheses group the patterns "abc", "def", and "ghi" together, and the pipe `|` acts as a logical OR operator, matching any of the grouped patterns.

## Task 2: Use regex groups to find and replace a pattern in a string

Regex groups can also be used to find and replace patterns in a string using the `re.sub()` function. The `\number` syntax is used to refer to groups in the replacement string, where `number` is the group number.

Consider the following example where we want to swap the first and last names in a string:

```python
import re

pattern = r"(\w+) (\w+)"
replacement = r"\2 \1"
text = "John Doe"
new_text = re.sub(pattern, replacement, text)

print(new_text)  # Output: 'Doe John'
```

In the pattern `r"(\w+) (\w+)"`, the first group `(\w+)` matches the first name and the second group `(\w+)` matches the last name. In the replacement string `r"\2 \1"`, `\2` refers to the second group (last name) and `\1` refers to the first group (first name), effectively swapping the names.

## Task 3: Create a regex to capture and extract phone numbers from text

Capturing groups can be used to extract information from text. Consider the following example where we want to extract phone numbers from a text:

```python
import re

pattern = r"(\d{3})-(\d{3})-(\d{4})"
text = "My phone number is 123-456-7890."
matches = re.search(pattern, text)

print(matches.groups())  # Output: ('123', '456', '7890')
```

In the pattern `r"(\d{3})-(\d{3})-(\d{4})"`, the first group `(\d{3})` matches the area code, the second group `(\d{3})` matches the first three digits, and the third group `(\d{4})` matches the last four digits of the phone number. The `re.search()` function returns a match object, and the `.groups()` method returns a tuple containing all the captured groups.

In conclusion, grouping and capturing in regex provide a powerful way to match complex patterns and extract information from text.