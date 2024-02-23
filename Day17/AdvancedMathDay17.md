# Day 17: Bayes' Theorem in Probability & Statistics

## Introduction
Bayes' Theorem is a key concept in probability and statistics. It's a way to update our beliefs or probabilities based on new data. This theorem is named after Thomas Bayes, who first provided an equation that allows new evidence to update beliefs.

## Task 1: Learn the formula and conceptual foundation of Bayes' Theorem

Bayes' Theorem is expressed as:

P(A|B) = [P(B|A) * P(A)] / P(B)

Where:
- P(A|B) is the probability of event A given event B is true.
- P(B|A) is the probability of event B given event A is true.
- P(A) and P(B) are the probabilities of events A and B respectively.

In simple words, it tells us how often A happens given that B happens, written P(A|B).

## Task 2: Understand the role of prior, likelihood, and posterior probabilities

In the context of Bayes' theorem, we often talk about prior, likelihood, and posterior probabilities.

- **Prior Probability (P(A))**: This is our initial belief about the probability of an event before new data is introduced.

- **Likelihood (P(B|A))**: This is the probability of the new data under our initial belief.

- **Posterior Probability (P(A|B))**: This is the updated belief that takes into account the new data.

The power of Bayes' theorem lies in its ability to provide a mathematical framework for updating our beliefs in the light of new data.

## Task 3: Practice solving problems involving Bayes' Theorem, including applications in decision-making

Let's consider a practical example. Suppose you are a doctor. You use a test to detect a disease. The disease affects 1% of the population. The test is 90% accurate. What is the probability that a person has the disease given that their test result is positive?

Here, we can apply Bayes' theorem:

- Prior Probability, P(Disease) = 0.01 (1% of the population have the disease)
- Likelihood, P(Positive|Disease) = 0.9 (The test is 90% accurate)
- P(Positive) = P(Disease) * P(Positive|Disease) + P(No Disease) * P(Positive|No Disease)
- P(Positive) = 0.01 * 0.9 + 0.99 * 0.1 = 0.108

Now, we can apply Bayes' theorem to find the posterior probability:

P(Disease|Positive) = (P(Positive|Disease) * P(Disease)) / P(Positive)
= (0.9 * 0.01) / 0.108
= 0.083

So, even if the test result is positive, there's only an 8.3% chance that the person actually has the disease. This is the power of Bayes' theorem - it allows us to update our beliefs (in this case, the belief that the patient has the disease) based on new data (the test result).

In conclusion, Bayes' theorem is a powerful tool in probability and statistics that allows us to update our beliefs based on new data. It has wide applications in various fields such as medicine, machine learning, decision-making, and more.