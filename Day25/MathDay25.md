# Day 25: Limits and Continuity

## Introduction
In the world of calculus, the concept of limits is a fundamental building block. It helps us understand the behavior of functions and allows us to solve complex mathematical problems. Alongside limits, we also have the concept of continuity, which tells us about the smoothness of a function. Today, we will dive into these concepts and learn how to calculate simple limits and understand the idea of continuity.

## Task 1: Calculate Simple Limits of Functions

A limit is a value that a function approaches as the input (or variable) approaches a certain value. In mathematical terms, we say that the limit of a function `f(x)` as `x` approaches `a` is `L` if `f(x)` can be made arbitrarily close to `L` by making `x` sufficiently close to `a`.

Let's consider a simple function `f(x) = x^2`. We want to find the limit as `x` approaches `2`. We can write this as `lim (x->2) x^2`.

To calculate this limit, we simply substitute `2` in place of `x` in the function. So, `lim (x->2) x^2 = 2^2 = 4`.

## Task 2: Understand the Concept of Continuity in a Function

A function is said to be continuous at a point if the limit of the function at that point exists and is equal to the value of the function at that point. In other words, a function `f(x)` is continuous at `x = a` if `lim (x->a) f(x) = f(a)`.

Let's consider the function `f(x) = x^2` again. We already calculated that `lim (x->2) x^2 = 4`. The value of the function at `x = 2` is also `4` (`f(2) = 2^2 = 4`). Since the limit of the function at `x = 2` is equal to the value of the function at `x = 2`, we can say that the function `f(x) = x^2` is continuous at `x = 2`.

## Task 3: Solve Problems Involving Limits and Continuity

Now that we understand the concepts of limits and continuity, let's solve a problem.

Consider the function `f(x) = (x^2 - 4) / (x - 2)`. We want to find the limit as `x` approaches `2`.

If we try to substitute `2` in place of `x` in the function, we get a division by zero, which is undefined. However, we can simplify the function by factoring the numerator as `(x - 2)(x + 2)`. The function then becomes `f(x) = (x + 2)`, for `x ≠ 2`.

Now, we can calculate the limit as `x` approaches `2`. `lim (x->2) (x + 2) = 2 + 2 = 4`.

Even though the function `f(x) = (x^2 - 4) / (x - 2)` is not defined at `x = 2`, the limit as `x` approaches `2` exists and is equal to `4`.

## Conclusion

In this lesson, we learned about the concept of limits and how to calculate simple limits of functions. We also learned about the idea of continuity and how it relates to limits. These concepts are fundamental in calculus and will be used in many mathematical problems.