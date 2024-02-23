# Python Control Structures

Control structures in Python are used to control the flow of execution in a program based on certain conditions or loops. They are essential for making decisions in code and for repeating blocks of code.

## Conditional Statements: if, elif, and else

Python supports the usual logical conditions from mathematics:

- Equals: `a == b`
- Not Equals: `a != b`
- Less than: `a < b`
- Less than or equal to: `a <= b`
- Greater than: `a > b`
- Greater than or equal to: `a >= b`

These conditions can be used in several ways, most commonly in "if statements" and loops.

An "if statement" is written by using the `if` keyword. Python relies on indentation (whitespace at the beginning of a line) to define scope in the code. Other programming languages often use curly-brackets for this purpose.

```python
a = 33
b = 200
if b > a:
  print("b is greater than a")
```

In this example we use two variables, `a` and `b`, which are used as part of the `if` statement to test whether `b` is greater than `a`. As `a` is 33, and `b` is 200, we know that 200 is greater than 33, and so we print to screen that "b is greater than a".

The `elif` keyword is pythons way of saying "if the previous conditions were not true, then try this condition".

```python
a = 33
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
```

The `else` keyword catches anything which isn't caught by the preceding conditions.

```python
a = 200
b = 33
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")
```

## Loops: for and while

Python has two primitive loop commands:

- `for` loops
- `while` loops

A `for` loop is used for iterating over a sequence (that is either a list, a tuple, a dictionary, a set, or a string).

```python
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)
```

The `for` loop does not require an indexing variable to set beforehand.

A `while` loop we can execute a set of statements as long as a condition is true.

```python
i = 1
while i < 6:
  print(i)
  i += 1
```

The `while` loop requires relevant variables to be ready, in this example we need to define an indexing variable, `i`, which we set to 1.

## Tasks

### Task 1: Write a Python program that uses if-elif-else statements to perform different actions based on multiple conditions.

```python
temperature = 35

if temperature > 30:
    print("It's a hot day.")
elif 20 <= temperature <= 30:
    print("It's a nice day.")
else:
    print("It's a cold day.")
```

### Task 2: Implement a for-loop that iterates over a list of numbers to calculate their cumulative sum.

```python
numbers = [1, 2, 3, 4, 5]
cumulative_sum = 0

for number in numbers:
    cumulative_sum += number
    print(f"Cumulative sum: {cumulative_sum}")
```

### Task 3: Use a while-loop to implement a guessing game that continues until the user guesses the correct number.

```python
correct_number = 7
guess = 0

while guess != correct_number:
    guess = int(input("Guess the number: "))
    if guess < correct_number:
        print("Too low!")
    elif guess > correct_number:
        print("Too high!")
    else:
        print("You got it!")
```