# Python Classes and Object-Oriented Programming

Object-Oriented Programming (OOP) is a programming paradigm that provides a means of structuring programs so that properties and behaviors are bundled into individual objects. Python, being a multi-paradigm programming language, supports OOP. In this tutorial, we will explore the principles of OOP in Python, focusing on classes, instances, and inheritance.

## Task 1: Define a Class Representing a Simple Geometric Shape

In Python, a class is defined using the `class` keyword. A class is a blueprint for creating objects (a particular data structure), providing initial values for state (member variables or attributes), and implementations of behavior (member functions or methods).

Let's define a simple class `Shape` that represents a geometric shape. This class will have two attributes: `name` and `area`.

```python
class Shape:
    def __init__(self, name, area):
        self.name = name
        self.area = area
```

The `__init__` method is a special method that Python calls when it creates a new instance of the class. It is used to initialize the attributes of the class.

## Task 2: Add an Instance Method to the Class

Instance methods are functions that are defined inside a class and can only be called from an instance of that class. Let's add an instance method `describe` to our `Shape` class that prints the name and area of the shape.

```python
class Shape:
    def __init__(self, name, area):
        self.name = name
        self.area = area

    def describe(self):
        return f"This is a {self.name} with an area of {self.area} square units."
```

## Task 3: Investigate Inheritance by Creating a Subclass

Inheritance is a way of creating a new class using details of an existing class without modifying it. The newly formed class is a derived class (or child class). Similarly, the existing class is a base class (or parent class).

Let's create a subclass `Circle` that inherits from the `Shape` class. A circle is a shape with additional properties, such as radius. We can calculate the area of a circle using the formula πr², where r is the radius of the circle.

```python
import math

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
        super().__init__("Circle", self.calculate_area())

    def calculate_area(self):
        return math.pi * (self.radius ** 2)
```

In the `Circle` class, we use the `super` function to call the `__init__` method of the `Shape` class, allowing us to use its functionality. The `calculate_area` method is specific to the `Circle` class and calculates the area of the circle.

This tutorial has introduced the basics of classes and object-oriented programming in Python. With these concepts, you can start to build more complex and structured programs.