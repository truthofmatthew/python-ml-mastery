# Day 9: Named Groups and Non-Capturing Groups in Regex

Regular expressions (regex) are a powerful tool for pattern matching and manipulation in text. They can be used to find, replace, and extract information from strings. In this tutorial, we will explore two advanced features of regex: named groups and non-capturing groups.

## Task 1: Write a regex that uses named groups to match dates

Named groups in regex allow us to assign a name to a group, which can make our regex more readable and easier to maintain. The syntax for named groups is `(?P<name>...)`, where `name` is the name of the group and `...` is the pattern.

Let's write a regex to match dates in the format `YYYY-MM-DD`:

```python
import re

# Define the regex pattern
pattern = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"

# Test the pattern
match = re.search(pattern, "The event will be held on 2022-05-01.")
if match:
    print(f"Year: {match.group('year')}")
    print(f"Month: {match.group('month')}")
    print(f"Day: {match.group('day')}")
```

In this example, we have three named groups: `year`, `month`, and `day`. We can access the matched values using these names.

## Task 2: Create a regex with non-capturing groups to match patterns without saving

Sometimes, we need to group parts of our regex pattern, but we don't want to save the matched values. This is where non-capturing groups come in handy. The syntax for non-capturing groups is `(?:...)`.

Let's write a regex to match phone numbers, but we only want to capture the area code:

```python
# Define the regex pattern
pattern = r"(?P<area_code>\d{3})-(?:\d{3}-\d{4})"

# Test the pattern
match = re.search(pattern, "My phone number is 123-456-7890.")
if match:
    print(f"Area code: {match.group('area_code')}")
```

In this example, the second group is a non-capturing group. It matches the pattern, but it doesn't save the matched value.

## Task 3: Combine named and non-capturing groups in a complex regex pattern

We can combine named and non-capturing groups in a single regex pattern. Let's write a regex to match email addresses, but we only want to capture the username and domain name:

```python
# Define the regex pattern
pattern = r"(?P<username>[\w.-]+)@(?:[\w.-]+\.)+(?P<domain>\w+)"

# Test the pattern
match = re.search(pattern, "My email address is john.doe@example.com.")
if match:
    print(f"Username: {match.group('username')}")
    print(f"Domain: {match.group('domain')}")
```

In this example, the second group is a non-capturing group that matches the pattern for the top-level domain (e.g., `.com`, `.net`, `.org`). The named groups `username` and `domain` capture the username and domain name, respectively.

In conclusion, named groups and non-capturing groups can make our regex patterns more flexible and easier to understand. They are powerful tools that can greatly enhance our text processing capabilities.