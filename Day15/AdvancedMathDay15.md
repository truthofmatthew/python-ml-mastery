# Day 15: Combinatorics and Counting

## Introduction
Combinatorics is a branch of mathematics that deals with counting, arranging, and combining objects. It is a key concept in computer science, especially in areas such as algorithm design, cryptography, and network design. 

## Task 1: Basic Counting Principles (Addition and Multiplication Principles)

### Addition Principle
The addition principle states that if there are `n` ways to do one thing and `m` ways to do another, and these two things cannot be done at the same time, then there are `n + m` ways to do either of the two things. 

For example, if you have 3 shirts and 2 pants, you can wear 3 different shirts or 2 different pants, but not both at the same time. So, there are `3 + 2 = 5` different ways to choose what to wear.

### Multiplication Principle
The multiplication principle states that if there are `n` ways to do one thing and `m` ways to do another, and these two things can be done at the same time, then there are `n * m` ways to do both. 

For example, if you have 3 shirts and 2 pants, you can wear one of the 3 shirts and one of the 2 pants at the same time. So, there are `3 * 2 = 6` different outfits you can wear.

## Task 2: Permutations and Combinations

### Permutations
A permutation is an arrangement of objects in a specific order. The number of permutations of `n` objects taken `r` at a time is given by `nPr = n! / (n-r)!`, where `n!` denotes the factorial of `n`.

For example, the number of ways to arrange 3 books on a shelf (taken all at a time) is `3P3 = 3! / (3-3)! = 6`.

### Combinations
A combination is a selection of objects without regard to the order of selection. The number of combinations of `n` objects taken `r` at a time is given by `nCr = n! / [(n-r)! * r!]`.

For example, the number of ways to choose 2 books from a set of 3 books is `3C2 = 3! / [(3-2)! * 2!] = 3`.

## Task 3: Combinatorial Problems Involving Binomial Coefficients

A binomial coefficient, often read as "n choose r", is the number of ways to choose `r` objects from a set of `n` objects without regard to order. It is given by `nCr = n! / [(n-r)! * r!]`.

For example, the number of ways to choose 2 people from a group of 5 people is `5C2 = 5! / [(5-2)! * 2!] = 10`.

Binomial coefficients are used in various combinatorial problems, such as counting subsets, combinations, and permutations, and they also appear in the binomial theorem, which gives the coefficients in the expansion of `(x + y)^n`.

In conclusion, combinatorics is a powerful tool for counting and arranging objects, and it has many applications in computer science and other fields. Understanding the basic principles of combinatorics and how to solve combinatorial problems is essential for anyone studying or working in these areas.