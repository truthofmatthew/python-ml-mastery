# Day 3: Limits and Continity in Calculus

## Introduction
In the world of calculus, limits and continuity are two very important concepts. They help us understand how values behave as they get closer to a certain point. 

## Task 1: Calculating Limits of Functions Algebraically

A limit is a value that a function or sequence "approaches" as the input (or index) "approaches" some value. In simple words, it's like saying, "What value is my function getting close to as my input gets close to a certain number?"

Let's take an example:

Consider the function `f(x) = x^2`. What is the limit as `x` approaches 2?

To find this, we simply substitute `x` with 2 in the function:

```
f(x) = x^2
f(2) = 2^2 = 4
```

So, the limit of the function `f(x) = x^2` as `x` approaches 2 is 4.

## Task 2: Understanding Continuity and Its Relation to Limits

A function is said to be continuous at a certain point if the limit exists at that point and is equal to the function's value at that point. In other words, there are no jumps, breaks, or holes in the function at that point.

Let's take the same function `f(x) = x^2`. Is it continuous at `x = 2`?

To check this, we need to see if the limit as `x` approaches 2 is the same as the function's value at `x = 2`. We already found that the limit as `x` approaches 2 is 4. The function's value at `x = 2` is also 4. So, the function `f(x) = x^2` is continuous at `x = 2`.

## Task 3: Solving Problems Related to Finding Limits and Testing Functions for Continuity

Now, let's try a slightly more complex problem.

Consider the function `f(x) = (x^2 - 4) / (x - 2)`. What is the limit as `x` approaches 2?

If we try to substitute `x` with 2 directly, we get a division by zero, which is undefined. But, we can simplify the function:

```
f(x) = (x^2 - 4) / (x - 2)
     = ((x - 2)(x + 2)) / (x - 2)
     = x + 2
```

Now, we can find the limit as `x` approaches 2:

```
f(2) = 2 + 2 = 4
```

So, the limit of the function `f(x) = (x^2 - 4) / (x - 2)` as `x` approaches 2 is 4.

However, the function is not defined at `x = 2` (because of the division by zero in the original function), so it is not continuous at `x = 2`.

## Conclusion

Limits and continuity are fundamental concepts in calculus. They help us understand how functions behave near certain points, and they are essential for more advanced topics in calculus. Practice finding limits and testing functions for continuity to get a good grasp of these concepts.