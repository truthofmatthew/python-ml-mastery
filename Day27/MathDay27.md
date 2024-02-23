# Day 27: Integration

## Introduction

Integration is a fundamental concept in calculus, often viewed as the reverse process of differentiation. While differentiation breaks things down, integration brings them together. It's used to calculate areas, volumes, central points, and many useful things. But it is easiest to start with finding the area under the curve of a function.

## Learning Objectives

By the end of this lesson, you should be able to:

1. Understand the concept of the antiderivative.
2. Integrate simple functions.
3. Apply integration to find areas under curves.

## Task 1: Understand the Concept of the Antiderivative

The antiderivative of a function `f(x)` is a function whose derivative is `f(x)`. In other words, if `F(x)` is an antiderivative of `f(x)`, then `F'(x) = f(x)`. 

For example, consider the function `f(x) = 2x`. The antiderivative of `f(x)` is `F(x) = x^2`, because the derivative of `x^2` is `2x`.

## Task 2: Integrate Simple Functions

The process of finding the antiderivative is called integration. The symbol for integration is ∫, and the function we want to integrate (in this case, `f(x)`) is written next to it. 

For example, to integrate the function `f(x) = x^2`, we write `∫x^2 dx`. The `dx` at the end tells us with respect to what variable we are integrating.

The integral of `x^2` is `(1/3)x^3 + C`, where `C` is the constant of integration. This means that any function of the form `(1/3)x^3 + C` is an antiderivative of `x^2`.

## Task 3: Apply Integration to Find Areas Under Curves

One of the main applications of integration is finding the area under a curve. If `f(x)` is a positive function, then the area under the curve of `f(x)` from `x = a` to `x = b` is given by the definite integral `∫ from a to b f(x) dx`.

For example, to find the area under the curve of `f(x) = x^2` from `x = 1` to `x = 2`, we calculate the definite integral `∫ from 1 to 2 x^2 dx`. This equals `[(1/3) * 2^3 - (1/3) * 1^3] = 7/3`.

In the next lesson, we will explore more applications of integration, including finding volumes and solving differential equations.