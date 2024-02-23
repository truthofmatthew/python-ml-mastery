# Day 2: Basic Probability Concepts

## Introduction
Probability is a way to measure how likely it is that some event will happen. It is a number between 0 and 1, where 0 means the event will not happen and 1 means it will. This is a key concept in computer science, especially in areas like machine learning and data analysis.

## Task 1: Sample Space, Events, and Probability Measures

### Sample Space
The sample space, often denoted by `S`, is the set of all possible outcomes of a random experiment. For example, if we roll a six-sided die, the sample space is `{1, 2, 3, 4, 5, 6}`.

### Events
An event is a set of outcomes from the sample space. For example, "rolling an even number" is an event for the dice experiment, and the set of outcomes is `{2, 4, 6}`.

### Probability Measures
A probability measure is a function that assigns probabilities to events. The probability of an event `E`, denoted by `P(E)`, is a number between 0 and 1. The sum of probabilities of all possible outcomes is 1.

## Task 2: Independent and Dependent Events

### Independent Events
Two events `A` and `B` are independent if the occurrence of `A` does not affect the probability of `B`, and vice versa. The probability of both `A` and `B` happening is the product of their individual probabilities: `P(A and B) = P(A) * P(B)`.

### Dependent Events
If the occurrence of `A` changes the probability of `B`, then `A` and `B` are dependent events. The probability of both `A` and `B` happening is: `P(A and B) = P(A) * P(B|A)`, where `P(B|A)` is the conditional probability of `B` given `A`.

## Task 3: Conditional Probability

Conditional probability is the probability of an event given that another event has occurred. If `A` and `B` are events, the conditional probability of `A` given `B` is denoted by `P(A|B)`, and is calculated as `P(A and B) / P(B)`, provided that `P(B) > 0`.

## Practice Problems

1. A bag contains 3 red balls and 2 blue balls. If a ball is drawn at random, what is the probability that it is red?

2. Two dice are rolled. What is the probability that the sum of the numbers is 7?

3. A card is drawn from a standard deck of 52 cards. What is the probability that it is a king given that it is a face card?

## Conclusion

Understanding probability is crucial in computer science. It forms the basis for statistical inference, which is used in data analysis and machine learning. By mastering these basic concepts, you are well on your way to becoming proficient in these areas.