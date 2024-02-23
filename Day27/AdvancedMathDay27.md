# Number Theory in Discrete Mathematics

## Introduction
Number theory is a branch of mathematics that deals with the properties and relationships of numbers, especially integers. It is a fundamental part of mathematics and has applications in computer science, cryptography, and more.

## Task 1: Divisibility, Prime Numbers, and Greatest Common Divisors

### Divisibility
In number theory, we say that an integer `a` divides another integer `b` if there exists an integer `c` such that `b = ac`. This is often written as `a|b`.

For example, `3` divides `6` because `6 = 3*2`.

### Prime Numbers
A prime number is a natural number greater than `1` that has no positive divisors other than `1` and itself. The first few prime numbers are `2, 3, 5, 7, 11, 13, 17, 19, 23, 29`.

### Greatest Common Divisors
The greatest common divisor (GCD) of two integers `a` and `b` is the largest number that divides both of them without leaving a remainder. 

For example, the GCD of `8` and `12` is `4`.

## Task 2: Modular Arithmetic and Its Properties

Modular arithmetic is a system of arithmetic for integers, where numbers "wrap around" after reaching a certain value—the modulus.

The notation `a ≡ b (mod m)` means that `a` is equivalent to `b` modulo `m`. This is true if `m` divides `a - b`.

For example, `17 ≡ 2 (mod 5)` because `5` divides `17 - 2`.

Properties of modular arithmetic include:

- `(a + b) mod m = ((a mod m) + (b mod m)) mod m`
- `(a - b) mod m = ((a mod m) - (b mod m)) mod m`
- `(a * b) mod m = ((a mod m) * (b mod m)) mod m`

## Task 3: Congruences and the Chinese Remainder Theorem

### Congruences
In number theory, two integers `a` and `b` are said to be congruent modulo `n` if `a ≡ b (mod n)`. This means that `a` and `b` leave the same remainder when divided by `n`.

### Chinese Remainder Theorem
The Chinese remainder theorem is a theorem of number theory, which states that if one knows the remainders of the division of an integer `n` by several integers, then one can determine uniquely the remainder of the division of `n` by the product of these integers, under certain conditions.

For example, if we know that `n ≡ 1 (mod 3)`, `n ≡ 2 (mod 4)`, and `n ≡ 3 (mod 5)`, then we can determine that `n ≡ 11 (mod 60)`.

## Conclusion
Number theory is a fascinating branch of mathematics with many practical applications. Understanding the basics of number theory, such as divisibility, prime numbers, and modular arithmetic, can provide a solid foundation for further study in fields like cryptography and computer science.