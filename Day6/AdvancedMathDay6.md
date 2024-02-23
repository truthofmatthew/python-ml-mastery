# Matrices and Matrix Operations

## Introduction
Matrices are like boxes that hold numbers. They are very important in a part of math called linear algebra. We can do many things with these boxes of numbers, like adding them, multiplying them, or even flipping them around!

## Task 1: Matrix Addition, Scalar Multiplication, and Matrix Multiplication

### Matrix Addition
Adding matrices is easy. You just add the numbers in the same positions in each matrix. Let's say we have two matrices A and B:

```
A = 1 2     B = 4 5
    3 4         6 7
```
To add A and B, we add the numbers in the same spots:

```
A + B = (1+4) (2+5) = 5 7
        (3+6) (4+7)   9 11
```

### Scalar Multiplication
Scalar multiplication is when we multiply a matrix by a single number (a scalar). If we have a matrix A and a scalar c:

```
A = 1 2     c = 3
    3 4
```
To multiply A by c, we multiply each number in A by c:

```
cA = 3*1 3*2 = 3 6
     3*3 3*4   9 12
```

### Matrix Multiplication
Matrix multiplication is a bit trickier. To multiply two matrices, the number of columns in the first matrix must be the same as the number of rows in the second. If we have two matrices A and B:

```
A = 1 2     B = 4 5
    3 4         6 7
```
To multiply A and B, we multiply each row in A by each column in B and add the results:

```
AB = (1*4 + 2*6) (1*5 + 2*7) = 16 19
     (3*4 + 4*6) (3*5 + 4*7)   36 43
```

## Task 2: Transpose of a Matrix and Its Properties

The transpose of a matrix is what you get when you flip the matrix over its diagonal. The diagonal is the line of numbers that starts at the top left and goes down to the bottom right. If we have a matrix A:

```
A = 1 2
    3 4
```
The transpose of A, written as A', is:

```
A' = 1 3
     2 4
```
The transpose of a matrix has some interesting properties. For example, the transpose of the transpose of a matrix is the original matrix (A'' = A), and the transpose of the sum of two matrices is the sum of their transposes ((A + B)' = A' + B').

## Task 3: Solving Linear Equations Using Matrices and Gaussian Elimination

Matrices can also be used to solve systems of linear equations. A system of linear equations is a set of equations that all need to be true at the same time. For example:

```
1x + 2y = 3
4x + 5y = 6
```
We can write this system as a matrix equation AX = B, where A is the matrix of coefficients, X is the matrix of variables, and B is the matrix of constants:

```
A = 1 2     X = x     B = 3
    4 5         y         6
```
To solve for X, we can use a method called Gaussian elimination. This method involves swapping rows, multiplying rows by scalars, and adding rows to each other to make the matrix A into an identity matrix (a matrix with 1s on the diagonal and 0s everywhere else). Once A is an identity matrix, X = B, and we have our solution!

That's it for today's lesson on matrices and matrix operations. Remember, practice makes perfect, so try some problems on your own to really understand these concepts!