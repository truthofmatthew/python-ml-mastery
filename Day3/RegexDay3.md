# Day 3: Quantifiers in Regex

Quantifiers in regex are used to specify the number of times a character, group, or character class must be present for a match to be found. They are an essential part of regex patterns, allowing us to match different amounts of the same character or group of characters.

## Task 1: Write a regex to find 'a' followed by one or more 'b's

To find an 'a' followed by one or more 'b's, we use the `+` quantifier. The `+` quantifier matches one or more of the preceding element.

```python
import re

# Define the string
text = "ab abb abbb abbbb"

# Define the regex pattern
pattern = "ab+"

# Find matches
matches = re.findall(pattern, text)

# Print matches
print(matches)  # Output: ['ab', 'abb', 'abbb', 'abbbb']
```

In the above example, the regex pattern `ab+` matches 'ab', 'abb', 'abbb', and 'abbbb' in the string.

## Task 2: Create a regex to match three digits in a row

To match exactly three digits in a row, we use the `{}` quantifier. The `{}` quantifier specifies exactly how many times the preceding character or group should match.

```python
import re

# Define the string
text = "123 4567 89 100 200 3000"

# Define the regex pattern
pattern = "\d{3}"

# Find matches
matches = re.findall(pattern, text)

# Print matches
print(matches)  # Output: ['123', '456', '100', '200', '300']
```

In the above example, the regex pattern `\d{3}` matches '123', '456', '100', '200', and '300' in the string.

## Task 3: Use a regex to find optional characters in a string

To find optional characters in a string, we use the `?` quantifier. The `?` quantifier matches zero or one of the preceding element, making it optional.

```python
import re

# Define the string
text = "color colour"

# Define the regex pattern
pattern = "colou?r"

# Find matches
matches = re.findall(pattern, text)

# Print matches
print(matches)  # Output: ['color', 'colour']
```

In the above example, the regex pattern `colou?r` matches both 'color' and 'colour' in the string. The 'u' is optional due to the `?` quantifier.