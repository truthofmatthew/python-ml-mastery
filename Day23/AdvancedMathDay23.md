# Vector Spaces and Subspaces in Linear Algebra

## Introduction

In linear algebra, a vector space (also called a linear space) is a collection of objects called vectors, which can be added together and multiplied ("scaled") by numbers, called scalars. Scalars are often taken to be real numbers, but there are also vector spaces with scalar multiplication by complex numbers, rational numbers, or generally any field.

## Task 1: Definitions and Properties of Vector Spaces and Subspaces

### Vector Spaces

A vector space is a set V on which two operations + and · are defined, called vector addition and scalar multiplication. The operation + (vector addition) must satisfy the following conditions:

- Closure: If u and v are vectors in V, then the sum u + v belongs to V.
- Associativity: u + (v + w) = (u + v) + w for all vectors u, v, w in V.
- Identity: There is a vector 0 in V such that 0 + v = v for all v in V.
- Inverse: For every vector v in V, there exists a vector -v in V such that v + (-v) = 0.

The operation · (scalar multiplication) must satisfy the following conditions:

- Closure: If v is a vector in V and c is a scalar, then the scalar product c · v belongs to V.
- Associativity: a · (b · v) = (a · b) · v for all scalars a, b and for all v in V.
- Identity: 1 · v = v for all v in V.
- Distributivity: a · (u + v) = a · u + a · v and (a + b) · v = a · v + b · v for all scalars a, b and for all vectors u, v in V.

### Subspaces

A subspace of a vector space is a set H of vectors for which the operations of addition and scalar multiplication are defined, and satisfy the following conditions:

- The zero vector of V is in H.
- H is closed under vector addition (that is, for every pair of vectors u and v in H, the sum u + v is also in H).
- H is closed under scalar multiplication (that is, for every vector u in H and every scalar c, the product c · u is in H).

## Task 2: Basis and Dimension of a Vector Space

The basis of a vector space V is a set of vectors that is linearly independent and spans V. This means that every vector in V can be written as a linear combination of the vectors in the basis. The number of vectors in a basis is called the dimension of the vector space.

## Task 3: Identifying Subspaces and Calculating Dimensions and Basis

To identify whether a set H is a subspace of a vector space V, we need to check if it satisfies the three conditions for a subspace. If it does, then H is a subspace of V.

To calculate the dimension of a vector space or a subspace, we need to find a basis for the space and count the number of vectors in the basis.

To find a basis for a vector space or a subspace, we need to find a set of vectors that is linearly independent and spans the space. This can be done by using the Gram-Schmidt process or by row reducing a matrix.