# Day 17: Visualization with Matplotlib and Seaborn

## Brief Introduction
Data visualization is a critical part of any data analysis process. It allows us to understand patterns, trends, and correlations that might not be obvious in text-based data. In Python, Matplotlib and Seaborn are two powerful libraries for creating a wide range of static, animated, and interactive plots.

## Learning Objective
By the end of this tutorial, you will be able to generate various types of visualizations to represent data using Matplotlib and Seaborn.

## Task 1: Use Matplotlib to Create a Line Chart Visualizing a Time Series Dataset

Matplotlib is a plotting library for Python. It provides an object-oriented API for embedding plots into applications.

Let's start by importing the necessary libraries and creating a simple line chart.

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a time series data
time = np.arange(0, 10, 0.1)
amplitude = np.sin(time)

# Create a line chart
plt.plot(time, amplitude)

# Add title and labels
plt.title('Sine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Display the plot
plt.show()
```

In the above code, we first create a time series data using numpy's `arange` and `sin` functions. Then, we use matplotlib's `plot` function to create a line chart. Finally, we add a title and labels to the chart using `title`, `xlabel`, and `ylabel` functions, and display the plot using `show` function.

## Task 2: Create a Seaborn Heatmap to Visualize Correlations within a Dataset

Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

Let's create a heatmap to visualize correlations within a dataset.

```python
import seaborn as sns
import pandas as pd

# Create a dataset
data = {'A': [45, 37, 42, 35, 39],
        'B': [38, 31, 26, 28, 33],
        'C': [10, 15, 17, 21, 12]}
df = pd.DataFrame(data)

# Calculate correlations
corr = df.corr()

# Create a heatmap
sns.heatmap(corr, annot=True)

# Display the plot
plt.show()
```

In the above code, we first create a pandas DataFrame from a dictionary. Then, we calculate the correlations between the columns using pandas' `corr` function. Finally, we create a heatmap using seaborn's `heatmap` function and display the plot.

## Task 3: Combine Multiple Plots into a Single Figure with Subplots

Matplotlib allows us to create multiple subplots in a single figure using the `subplot` function.

```python
# Create a new figure
plt.figure()

# Create the first subplot
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
plt.plot(time, amplitude)
plt.title('Sine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Create the second subplot
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
plt.plot(time, np.cos(time))
plt.title('Cosine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude')

# Adjust the layout
plt.tight_layout()

# Display the plots
plt.show()
```

In the above code, we first create a new figure using `figure` function. Then, we create two subplots using `subplot` function. The first argument is the number of rows, the second argument is the number of columns, and the third argument is the index of the current plot. Finally, we adjust the layout using `tight_layout` function and display the plots.

That's it for today's tutorial. You should now be able to create various types of visualizations using Matplotlib and Seaborn. Happy plotting!