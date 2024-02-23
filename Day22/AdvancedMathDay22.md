# Probability Density Functions and Distributions

## Introduction
Probability density functions (PDFs) are mathematical functions that describe the likelihood of different outcomes in an experiment or process. They are used in statistics and probability theory to model and analyze continuous random variables.

## Task 1: Understanding the Properties of Probability Density Functions

A probability density function has two key properties:

1. **Non-negativity**: The value of the PDF is always greater than or equal to zero. This means that the probability of any event is never negative.

2. **Normalization**: The total area under the curve of the PDF is equal to 1. This means that the sum of the probabilities of all possible outcomes is 1.

Let's consider an example. Suppose we have a random variable X that represents the height of a person. The PDF of X might look something like this:

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 170 # mean height
sigma = 10 # standard deviation
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.show()
```

In this plot, the x-axis represents the height of a person and the y-axis represents the probability density. The curve shows that most people have a height around 170 cm, and the probability decreases as the height deviates from this mean.

## Task 2: Common Continuous Distributions

There are many types of continuous distributions. Here, we will focus on two common ones: the normal distribution and the exponential distribution.

### Normal Distribution

The normal distribution, also known as the Gaussian distribution, is a bell-shaped curve that is symmetric around its mean. It is defined by two parameters: the mean (μ) and the standard deviation (σ). The PDF of a normal distribution is given by:

![Normal Distribution Formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/187f33664a6a5e2e8b864f4b1631e1022e6f6b0b)

### Exponential Distribution

The exponential distribution describes the time between events in a Poisson point process, i.e., a process in which events occur continuously and independently at a constant average rate. It has a single parameter λ, which is the rate of events. The PDF of an exponential distribution is given by:

![Exponential Distribution Formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/2f6526be5d3dec289f3a48735b4c7b5f7e3e7cda)

## Task 3: Calculating Probabilities and Expected Values for Continuous Distributions

The probability of an event in a continuous distribution is given by the area under the PDF curve over the range of the event. This is calculated using the integral of the PDF.

The expected value (or mean) of a continuous random variable is calculated by multiplying each possible outcome by its probability, and then summing these products. For a normal distribution, the expected value is equal to the mean μ. For an exponential distribution, the expected value is equal to 1/λ.

Let's calculate the probability that a person's height is between 160 and 180 cm, and the expected height, using the normal distribution from our earlier example:

```python
from scipy.integrate import quad

# Probability that height is between 160 and 180 cm
prob = quad(lambda x: stats.norm.pdf(x, mu, sigma), 160, 180)[0]
print(f"Probability: {prob}")

# Expected height
expected_height = mu
print(f"Expected height: {expected_height} cm")
```

In this code, `quad` is a function that calculates the integral of a function. We use it to calculate the area under the PDF curve between 160 and 180 cm. The expected height is simply the mean μ of the distribution.

Remember, understanding probability density functions and distributions is key to many areas of computer science, including machine learning, data analysis, and algorithm design. Keep practicing and exploring different types of distributions to strengthen your understanding.