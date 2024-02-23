# Introduction to Vectors in Linear Algebra

## Brief Introduction
Vectors are a key part of linear algebra. They are a way to show amounts that have both size and direction. 

## Learning Objective
Our goal is to understand basic vector actions and traits.

## Task 1: Vector Addition, Scalar Multiplication, and Their Properties

### Vector Addition
When we add two vectors, we join them head-to-tail and draw the result from the tail of the first to the head of the second. For example, if we have vectors `a = [2, 3]` and `b = [1, 2]`, the sum `a + b` would be `[2+1, 3+2] = [3, 5]`.

### Scalar Multiplication
When we multiply a vector by a scalar (a single number), we change its size but not its direction. For example, if we have a vector `a = [2, 3]` and we multiply it by the scalar `2`, the result would be `2*a = [2*2, 2*3] = [4, 6]`.

### Properties
Vector addition and scalar multiplication have several properties, including:

1. **Commutativity**: `a + b = b + a`
2. **Associativity**: `(a + b) + c = a + (b + c)`
3. **Distributivity**: `a * (b + c) = a*b + a*c`
4. **Identity**: There exists a zero vector `0` such that `a + 0 = a`
5. **Inverse**: For every vector `a`, there exists a vector `-a` such that `a + (-a) = 0`

## Task 2: Dot Products and Their Geometric Interpretation

The dot product of two vectors is a number that tells us about the angle between the vectors. If `a = [a1, a2]` and `b = [b1, b2]`, then the dot product `a . b` is `a1*b1 + a2*b2`.

The dot product has a geometric interpretation: if the dot product of two vectors is zero, the vectors are perpendicular. If the dot product is positive, the angle between the vectors is less than 90 degrees. If the dot product is negative, the angle is more than 90 degrees.

## Task 3: Vector Norms and Unit Vectors

The norm of a vector is its length. For a vector `a = [a1, a2]`, the norm `||a||` is `sqrt(a1^2 + a2^2)`.

A unit vector is a vector of length 1. To make a unit vector, we divide a vector by its norm. For example, if `a = [2, 3]`, the unit vector in the direction of `a` would be `a/||a|| = [2/sqrt(13), 3/sqrt(13)]`.

That's it for today's lesson! Practice these concepts and we'll explore more about vectors in the next lesson.