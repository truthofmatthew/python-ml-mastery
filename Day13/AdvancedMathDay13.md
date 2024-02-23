# Day 13: Integrals and Integration Techniques

## Introduction
Integration is a fundamental concept in calculus. It is the process of adding up all the small data points to find the total. In other words, it is the reverse process of differentiation. There are two types of integrals: indefinite and definite. 

## Task 1: Indefinite and Definite Integrals

### Indefinite Integrals
An indefinite integral, also known as an antiderivative, represents a family of functions. It is the reverse process of differentiation and is represented as ∫f(x) dx. The result of an indefinite integral is a function (or family of functions).

Example:
The indefinite integral of f(x) = 2x is F(x) = x² + C, where C is the constant of integration.

### Definite Integrals
A definite integral has upper and lower limits on the integral sign. It represents the exact area under a curve from one point to another and is represented as ∫ from a to b f(x) dx.

Example:
The definite integral of f(x) = 2x from 0 to 2 is ∫ from 0 to 2 2x dx = [x²] from 0 to 2 = 4 - 0 = 4.

## Task 2: Methods of Integration

### Substitution
The method of substitution is used when an integral contains a function and its derivative. We substitute a new variable for the function and its derivative.

Example:
To integrate ∫x(1 - x²)² dx, let u = 1 - x². Then du = -2x dx. The integral becomes -1/2 ∫u² du = -1/2 * (u³/3) + C = -1/6 (1 - x²)³ + C.

### Integration by Parts
Integration by parts is used when an integral is the product of two functions, one of which is easier to integrate, and the other, easier to differentiate. It is based on the rule of differentiation for the product of two functions.

Example:
To integrate ∫x e^x dx, let u = x (easier to differentiate) and dv = e^x dx (easier to integrate). Then du = dx and v = e^x. The integral becomes uv - ∫v du = x e^x - ∫e^x dx = x e^x - e^x + C.

## Task 3: Applications of Integration

### Area Under a Curve
The definite integral of a function can be used to calculate the area under the curve of that function.

Example:
The area under the curve of f(x) = x² from 0 to 2 is ∫ from 0 to 2 x² dx = [x³/3] from 0 to 2 = 8/3 - 0 = 8/3.

### Solving Physical Problems
Integration can be used to solve problems in physics such as finding the distance traveled by an object given its speed, or the work done by a force.

Example:
If a car travels at a speed of v(t) = 2t (miles per hour), the distance it travels from t = 0 to t = 2 hours is ∫ from 0 to 2 v(t) dt = ∫ from 0 to 2 2t dt = [t²] from 0 to 2 = 4 - 0 = 4 miles.