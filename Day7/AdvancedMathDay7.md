# Day 7: Random Variables and Probability Distributions

## Introduction
Random variables are a way to assign numbers to each possible outcome of a random process. They are a key concept in probability and statistics. Probability distributions describe how these random variables are spread out or distributed.

## Task 1: Discrete and Continuous Random Variables

### Discrete Random Variables
A discrete random variable is one that can take on a countable number of distinct values. For example, if you roll a die, the possible outcomes are 1, 2, 3, 4, 5, or 6. These are discrete values.

```python
import random
# Rolling a die
die_roll = random.randint(1, 6)
print(die_roll)
```

### Continuous Random Variables
A continuous random variable is one that can take on an infinite number of values within a certain range. For example, the height of a person is a continuous variable because it can be any value within a certain range.

```python
import numpy as np
# Generating a random height in meters
height = np.random.uniform(1.5, 2)
print(height)
```

## Task 2: Common Probability Distributions

### Binomial Distribution
A binomial distribution is a discrete probability distribution of the number of successes in a sequence of n independent experiments.

```python
from scipy.stats import binom
# Generating a binomial distribution
n, p = 10, 0.5  # number of trials, probability of each trial
rv = binom(n, p)
print(rv.pmf(5))  # probability mass function at 5
```

### Normal Distribution
A normal distribution is a type of continuous probability distribution for a real-valued random variable. It is also known as the Gaussian distribution.

```python
from scipy.stats import norm
# Generating a normal distribution
rv = norm(loc=0, scale=1)  # mean=0, standard deviation=1
print(rv.pdf(0))  # probability density function at 0
```

## Task 3: Calculating Mean, Variance, and Standard Deviation

### Mean
The mean is the average of a set of numbers. It is calculated by adding up all the numbers and then dividing by the count of numbers.

```python
numbers = [1, 2, 3, 4, 5]
mean = sum(numbers) / len(numbers)
print(mean)
```

### Variance
The variance is a measure of how spread out a distribution is. It is calculated by taking the average of the squared differences from the mean.

```python
variance = sum((n - mean) ** 2 for n in numbers) / len(numbers)
print(variance)
```

### Standard Deviation
The standard deviation is a measure of the amount of variation or dispersion of a set of values. It is the square root of the variance.

```python
std_dev = variance ** 0.5
print(std_dev)
```

In the next lesson, we will learn about more advanced topics in probability and statistics.