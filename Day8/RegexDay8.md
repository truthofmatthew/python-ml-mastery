# Backreferences in Regex

Backreferences in regular expressions allow us to refer back to groups that have already been matched. This can be particularly useful when we want to find repeated words or patterns within a string. In Python, we can use the `\1`, `\2`, etc., syntax to refer to the first, second, etc., matched group.

## Task 1: Match Repeated Words in a String Using Backreferences

Let's say we have a string and we want to find all instances of repeated words. We can use a regular expression with a backreference to do this.

```python
import re

# Our input string
s = "The the quick brown fox jumps over the lazy dog dog."

# Our regex pattern
p = r"\b(\w+)\b\s+\1\b"

# Find all matches
matches = re.findall(p, s, re.IGNORECASE)

# Print matches
print(matches)  # Output: ['the', 'dog']
```

In the above code, `\b(\w+)\b` matches a word (a sequence of one or more word characters surrounded by word boundaries), and `\s+\1\b` matches one or more spaces followed by the same word again.

## Task 2: Use a Backreference to Match Opening and Closing HTML Tags

Backreferences can also be used to match pairs of opening and closing HTML tags.

```python
# Our input string
s = "<title>The Title</title>"

# Our regex pattern
p = r"<(\w+)>(.*?)</\1>"

# Find all matches
matches = re.findall(p, s)

# Print matches
print(matches)  # Output: [('title', 'The Title')]
```

In the above code, `<(\w+)>` matches an opening HTML tag, `(.*?)` matches any content within the tag, and `</\1>` matches the corresponding closing tag.

## Task 3: Include a Backreference in a Search and Replace Function

We can also use backreferences in a search and replace function to replace matched groups with modified versions of themselves.

```python
# Our input string
s = "The the quick brown fox jumps over the lazy dog dog."

# Our regex pattern
p = r"\b(\w+)\b\s+\1\b"

# Replace matched groups with a single instance of the word
s = re.sub(p, r"\1", s, flags=re.IGNORECASE)

# Print the modified string
print(s)  # Output: "The quick brown fox jumps over the lazy dog."
```

In the above code, `re.sub(p, r"\1", s, flags=re.IGNORECASE)` replaces each matched group with a single instance of the word. The `\1` in the replacement string refers to the first matched group.