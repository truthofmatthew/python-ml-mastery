# Systems of Linear Equations

## Introduction
A system of linear equations is a collection of one or more linear equations involving the same variables. For example, with two variables x and y, a system of linear equations might look like this:

```
x + y = 5
2x - 3y = -1
```

In this tutorial, we will learn how to represent these systems as matrix equations and how to solve them using techniques such as Cramer's Rule and matrix inversion.

## Task 1: Representation of Linear Systems as Matrix Equations

A system of linear equations can be represented as a matrix equation. A matrix is a rectangular array of numbers arranged in rows and columns. For example, the system of equations above can be written as a matrix equation as follows:

```
[1 1] [x]   = [5]
[2 -3] [y]   = [-1]
```

Here, the matrix [1 1; 2 -3] is called the coefficient matrix, and the matrix [x; y] is called the variable matrix. The matrix on the right side [5; -1] is the constant matrix.

## Task 2: Solution Techniques - Cramer's Rule and Matrix Inversion

### Cramer's Rule
Cramer's Rule is a method used to solve systems of linear equations by expressing the solution in terms of the determinants of the coefficient matrix and a matrix created by replacing one column of the coefficient matrix with the constant matrix.

For the system of equations above, the determinant of the coefficient matrix is (1*-3) - (1*2) = -5. 

To find the value of x, we replace the first column of the coefficient matrix with the constant matrix and calculate the determinant:

```
[5 1]  -> determinant = (5*-3) - (1*-1) = -14
[-1 -3]
```

Then, x = determinant of the new matrix / determinant of the coefficient matrix = -14 / -5 = 2.8.

We can find the value of y in a similar way.

### Matrix Inversion
Another method to solve systems of linear equations is by using matrix inversion. The inverse of a matrix A is a matrix A^-1 such that when A is multiplied by A^-1, the result is the identity matrix. 

If we have a system of equations Ax = b, where A is the coefficient matrix, x is the variable matrix, and b is the constant matrix, then x = A^-1 * b.

## Task 3: Application of Techniques

Let's apply these techniques to solve a practical problem. Suppose we have a system of equations representing a shop where 1 apple and 1 banana cost $5, and 2 apples and 3 bananas cost $11. We can represent this system as a matrix equation:

```
[1 1] [apple]   = [5]
[2 3] [banana]  = [11]
```

Using Cramer's Rule or matrix inversion, we can find the cost of one apple and one banana.

## Conclusion
In this tutorial, we learned how to represent systems of linear equations as matrix equations and how to solve them using Cramer's Rule and matrix inversion. These techniques are fundamental in many areas of computer science, including graphics, machine learning, and optimization.