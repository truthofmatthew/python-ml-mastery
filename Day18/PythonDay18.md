# Day 18: Introduction to Data Science with Python

## Brief Introduction
Data science is a field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data. Python, with its rich ecosystem of libraries and tools, is one of the most popular languages for data science. In this tutorial, we will cover the basics of data science workflow using Python.

## Learning Objective
By the end of this tutorial, you will understand the data science workflow and how to use Python tools to load, clean, and analyze datasets.

## Prerequisites
- Basic knowledge of Python
- Familiarity with Pandas, NumPy, and Matplotlib libraries

---

## Task 1: Explore a Dataset

The first step in any data science project is to understand the dataset. We will use the Pandas library to load and explore a dataset.

```python
# Import the pandas library
import pandas as pd

# Load a dataset
df = pd.read_csv('your_dataset.csv')

# Display the first 5 rows of the dataset
print(df.head())

# Display the shape of the dataset
print('Shape of the dataset:', df.shape)

# Display the columns in the dataset
print('Columns in the dataset:', df.columns)
```

---

## Task 2: Data Cleaning

Data cleaning is a crucial step in the data science workflow. It involves handling missing values and cleaning text data.

```python
# Check for missing values
print(df.isnull().sum())

# Fill missing values with the mean of the column
df.fillna(df.mean(), inplace=True)

# Remove special characters from a text column
df['text_column'] = df['text_column'].str.replace('[^\w\s]','')
```

---

## Task 3: Exploratory Data Analysis (EDA)

EDA involves generating descriptive statistics and visualizing distributions for each column in your dataset.

```python
# Generate descriptive statistics
print(df.describe())

# Import the matplotlib library for visualization
import matplotlib.pyplot as plt

# Plot a histogram for a column
plt.hist(df['column_name'], bins=10, alpha=0.5)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Histogram of Column Name')
plt.show()
```

---

In this tutorial, we have covered the basics of the data science workflow using Python. We have learned how to load a dataset using Pandas, clean the dataset by handling missing values and cleaning text data, and perform exploratory data analysis by generating descriptive statistics and visualizing distributions.