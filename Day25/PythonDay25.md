# Day 25: Regular Expressions in Python - Advanced

In this tutorial, we will delve deeper into the world of regular expressions in Python. Regular expressions, or regex, are a powerful tool for manipulating text data. They allow us to match complex patterns within strings, extract useful information, and perform sophisticated replacements. We will be using Python's built-in `re` module for this tutorial.

## Task 1: Develop a regex pattern that matches complex URLs within text content and extract components (protocol, domain, path)

Let's start by developing a regex pattern that can match URLs within a string. A typical URL consists of a protocol (http, https), a domain, and a path. Here is an example of a URL: `https://www.example.com/path/to/page`.

```python
import re

# Define the regex pattern for URLs
url_pattern = r"(https?://)([^/]+)(/.*)?"

# Test the pattern on a string
text = "Visit https://www.example.com/path/to/page for more information."
matches = re.findall(url_pattern, text)

for match in matches:
    print(f"Protocol: {match[0]}")
    print(f"Domain: {match[1]}")
    print(f"Path: {match[2]}")
```

In the above code, we define a regex pattern for URLs and use it to find all matches in a string. The pattern is divided into three groups, each enclosed in parentheses. The first group `(https?://)` matches the protocol, the second group `([^/]+)` matches the domain, and the third group `(/.*)?` matches the path.

## Task 2: Create a regular expression to sanitize and replace sensitive user data (e.g., emails, phone numbers) within a string with obfuscated characters

Next, let's create a regex pattern to sanitize sensitive user data such as emails and phone numbers. We will replace these with obfuscated characters to protect user privacy.

```python
# Define the regex patterns for emails and phone numbers
email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
phone_pattern = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"

# Test the patterns on a string
text = "Contact John Doe at johndoe@example.com or 123-456-7890."
text = re.sub(email_pattern, "********@*****.***", text)
text = re.sub(phone_pattern, "************", text)

print(text)
```

In the above code, we define regex patterns for emails and phone numbers and use them to replace all matches in a string with obfuscated characters.

## Task 3: Utilize regex groups and backreferences to reformat date strings from one format (e.g., "YYYY-MM-DD") to another (e.g., "MM/DD/YYYY")

Finally, let's use regex groups and backreferences to reformat date strings. We will convert dates from the format "YYYY-MM-DD" to "MM/DD/YYYY".

```python
# Define the regex pattern for dates
date_pattern = r"(\d{4})-(\d{2})-(\d{2})"

# Test the pattern on a string
text = "The event will take place on 2022-12-31."
text = re.sub(date_pattern, r"\2/\3/\1", text)

print(text)
```

In the above code, we define a regex pattern for dates and use it to replace all matches in a string with a new format. The pattern is divided into three groups, each enclosed in parentheses. The first group `(\d{4})` matches the year, the second group `(\d{2})` matches the month, and the third group `(\d{2})` matches the day. In the replacement string, we use backreferences `\2`, `\3`, and `\1` to refer to these groups in a new order.

In conclusion, regular expressions are a powerful tool for text manipulation in Python. They allow us to match complex patterns, extract useful information, and perform sophisticated replacements. With practice, you can use them to solve a wide range of problems involving text data.