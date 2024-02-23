# Day 24: Advanced Python - Metaprogramming

Metaprogramming is a powerful feature of Python that allows you to manipulate the structure and behavior of your code at runtime. It enables you to create classes and functions dynamically, and to modify the attributes of objects. This tutorial will guide you through the process of using metaprogramming in Python.

## Task 1: Use the type function to dynamically create a new class at runtime

In Python, classes are objects too. You can create a new class at runtime using the built-in `type` function. The `type` function takes three arguments: the name of the class, a tuple of the parent classes (for inheritance), and a dictionary containing attributes and methods.

Here is an example:

```python
# Define a function that will be a method of the class
def say_hello(self):
    return f"Hello, I am {self.name}"

# Create a new class at runtime
Person = type('Person', (), {'say_hello': say_hello})

# Create an instance of the class
p = Person()
p.name = "John"
print(p.say_hello())  # Outputs: Hello, I am John
```

## Task 2: Implement a class decorator that adds new methods to the class dynamically

A decorator is a function that takes another function or class and extends its behavior without explicitly modifying it. You can use a class decorator to add new methods to the class dynamically.

Here is an example:

```python
def add_greeting(cls):
    # Define a new method
    def greeting(self):
        return f"Hello, I am {self.name}"
    
    # Add the method to the class
    setattr(cls, 'greeting', greeting)
    
    return cls

# Apply the decorator to the class
@add_greeting
class Person:
    def __init__(self, name):
        self.name = name

# Create an instance of the class
p = Person("John")
print(p.greeting())  # Outputs: Hello, I am John
```

## Task 3: Explore the getattr, setattr, and delattr built-in functions to manipulate object attributes dynamically

Python provides several built-in functions to manipulate the attributes of objects dynamically:

- `getattr(object, name[, default])` returns the value of the named attribute of `object`. If the attribute does not exist, it returns `default` if provided, otherwise it raises an `AttributeError`.

- `setattr(object, name, value)` sets the value of the named attribute of `object`, creating a new attribute if it does not exist.

- `delattr(object, name)` deletes the named attribute of `object`. If the attribute does not exist, it raises an `AttributeError`.

Here is an example:

```python
class Person:
    def __init__(self, name):
        self.name = name

p = Person("John")

# Get an attribute
print(getattr(p, 'name'))  # Outputs: John

# Set an attribute
setattr(p, 'age', 30)
print(p.age)  # Outputs: 30

# Delete an attribute
delattr(p, 'age')
print(hasattr(p, 'age'))  # Outputs: False
```

In this tutorial, you have learned how to use metaprogramming in Python to create classes and functions dynamically, and to manipulate the attributes of objects. This can be a powerful tool for writing flexible and reusable code.