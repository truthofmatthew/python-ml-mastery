# Advanced Regex: Lookahead and Lookbehind Assertions

In this tutorial, we will delve into the advanced aspects of regular expressions (regex) - lookahead and lookbehind assertions. These assertions allow us to match a pattern based on what comes before or after it, without including the lookahead or lookbehind pattern in the match.

## Task 1: Use regex lookahead to match a pattern followed by another

Lookahead assertions are used when we want to match a pattern only if it is followed by another pattern. The syntax for a positive lookahead is `(?=...)`, where `...` is the pattern we are looking for.

Let's consider an example where we want to match the word "Python" only if it is followed by the word "programming". 

```python
import re

text = "Python programming is fun. Python snake is not programming."
pattern = r"Python(?=\sprogramming)"

matches = re.findall(pattern, text)
print(matches)  # Output: ['Python']
```

In the above code, `(?=\sprogramming)` is a positive lookahead assertion. It checks if "Python" is followed by " programming", but does not include " programming" in the match.

## Task 2: Create a regex with lookbehind to find a pattern preceded by another

Lookbehind assertions are used when we want to match a pattern only if it is preceded by another pattern. The syntax for a positive lookbehind is `(?<=...)`.

Let's consider an example where we want to match the word "programming" only if it is preceded by the word "Python".

```python
text = "Python programming is fun. Snake programming is not Python."
pattern = r"(?<=Python\s)programming"

matches = re.findall(pattern, text)
print(matches)  # Output: ['programming']
```

In the above code, `(?<=Python\s)` is a positive lookbehind assertion. It checks if "programming" is preceded by "Python ", but does not include "Python " in the match.

## Task 3: Use lookahead and lookbehind together in a regex pattern

We can use lookahead and lookbehind assertions together in a regex pattern to match a pattern based on what comes before and after it.

Let's consider an example where we want to match the word "programming" only if it is preceded by the word "Python" and followed by the word "fun".

```python
text = "Python programming is fun. Snake programming is not Python."
pattern = r"(?<=Python\s)programming(?=\sis)"

matches = re.findall(pattern, text)
print(matches)  # Output: ['programming']
```

In the above code, `(?<=Python\s)` is a positive lookbehind assertion that checks if "programming" is preceded by "Python ", and `(?=\sis)` is a positive lookahead assertion that checks if "programming" is followed by " is". The word "programming" is matched only if both conditions are met.

In conclusion, lookahead and lookbehind assertions provide a powerful way to match patterns based on their context, without including the context in the match. They are an essential tool for advanced regex tasks.