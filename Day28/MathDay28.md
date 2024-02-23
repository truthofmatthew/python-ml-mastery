# Day 28: Matrices

## Brief Introduction: Basics of Matrix Operations

A matrix is a rectangular array of numbers arranged in rows and columns. It is a fundamental concept in mathematics and computer science, used in areas such as linear algebra, computer graphics, and machine learning. 

## Learning Objective: Understand and Perform Basic Operations with Matrices

By the end of this lesson, you should be able to define a matrix, understand its components, and perform basic operations such as addition, subtraction, scalar multiplication, and matrix multiplication.

---

### Task 1: Define a Matrix and Understand Its Components

A matrix is defined by its number of rows and columns. For example, a matrix with 2 rows and 3 columns is called a 2x3 matrix. Each number in the matrix is called an element. Here is an example of a 2x3 matrix:

```
1 2 3
4 5 6
```

In this matrix, the element in the first row and first column is 1, the element in the first row and second column is 2, and so on.

---

### Task 2: Perform Matrix Addition, Subtraction, and Scalar Multiplication

Matrix addition and subtraction are straightforward: you simply add or subtract the corresponding elements in the two matrices. The matrices must have the same dimensions (i.e., the same number of rows and columns) to be added or subtracted. Here is an example:

```
Matrix A:    Matrix B:    A + B:

1 2 3        4 5 6        5  7  9
4 5 6   +    7 8 9   =    11 13 15
```

Scalar multiplication involves multiplying every element in the matrix by a scalar (a single number). Here is an example with the scalar 2:

```
Matrix A:    2 * A:

1 2 3        2  4  6
4 5 6   *2 = 8 10 12
```

---

### Task 3: Multiply Matrices

Matrix multiplication is more complex than the previous operations. To multiply two matrices, the number of columns in the first matrix must be equal to the number of rows in the second matrix. 

The result of the multiplication is a new matrix where each element is the sum of the products of elements from the corresponding row of the first matrix and column of the second matrix. Here is an example:

```
Matrix A:    Matrix B:    A * B:

1 2          5 6          19 22
3 4     *    7 8     =    43 50
```

In the resulting matrix, the element in the first row and first column is (1*5 + 2*7) = 19, and the element in the first row and second column is (1*6 + 2*8) = 22, and so on.

---

Remember, practice is key in mastering these operations. Try to perform these operations with different matrices until you feel comfortable with the process.