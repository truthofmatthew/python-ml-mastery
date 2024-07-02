### Customer Analytics Presentation

#### 1. STP (Segmentation, Targeting, Positioning)

**Segmentation**
- **Demographic**: Age, gender, income level, education, occupation
- **Geographic**: Region, city size, population density, climate
- **Psychographic**: Lifestyle, social class, personality traits
- **Behavioral**: Purchase frequency, brand loyalty, user status, purchase occasion

*Example Data:*
- **Demographic**: Age 25-35, male, income $50,000-$70,000, college-educated
- **Geographic**: Urban areas, cities with populations over 500,000
- **Psychographic**: Health-conscious, tech-savvy, environmentally aware
- **Behavioral**: Buys organic products weekly, prefers brands with eco-friendly packaging

**Targeting**
- **Evaluating Segments**: Segment size, expected growth, competitive intensity, potential profitability
- **Selected Segment**: Urban millennials with high disposable income, environmentally conscious

*Example Data:*
- **Segment Size**: 2 million potential customers
- **Expected Growth**: 5% annual increase
- **Competitive Offering**: Competitors offer similar eco-friendly products but at higher prices

**Positioning**
- **Product Characteristics**: Eco-friendly, high-quality, innovative
- **Presentation to User**: Highlight sustainability, premium quality, and unique features
- **Channels**: Online marketplaces, social media, eco-friendly stores

*Example Data:*
- **Product Features**: Made from recycled materials, biodegradable packaging, innovative design
- **Presentation**: Ads showcasing environmental impact, influencer endorsements
- **Channels**: Amazon, Instagram, specialty eco-stores

#### 2. Elasticity
- **Purchase Probability**: Likelihood of a customer buying a product
- **Brand Choice**: Preference for a specific brand over others
- **Purchase Quantity**: Amount of product purchased per transaction

*Example Data:*
- **Purchase Probability**: 70% for eco-friendly product among target segment
- **Brand Choice**: 60% prefer Brand A over Brand B due to better sustainability practices
- **Purchase Quantity**: Average purchase is 2 units per transaction

#### 3. KYC (Know Your Customer)
- **Consumer Behavior Data**: Historical data, purchase frequency, time of purchase, product ratings

*Example Data:*
- **Historic Data**: 80% of purchases made online
- **Purchase Frequency**: Monthly repeat purchases by 50% of customers
- **Time of Purchase**: Peak sales during weekends
- **Product Ratings**: Average rating of 4.5 stars

#### 4. Marketing Mix
- **Product**: Characteristics and features
- **Price**: Offering and discounts
- **Promotion**: Communication, ads, sales
- **Place**: Distribution channels

*Example Data:*
- **Product**: Durable, eco-friendly, innovative design
- **Price**: Mid-range pricing, occasional discounts for loyal customers
- **Promotion**: Social media campaigns, influencer collaborations
- **Place**: Available on e-commerce platforms, selected retail stores

### Example Data Presentation

| **Segmentation** | **Data** |
|------------------|----------|
| **Demographic**  | Age: 25-35, Male, Income: $50,000-$70,000, Education: College |
| **Geographic**   | Urban areas, Cities > 500,000 population |
| **Psychographic**| Health-conscious, Tech-savvy, Environmentally aware |
| **Behavioral**   | Buys organic products weekly, Prefers eco-friendly brands |

| **Targeting**    | **Data** |
|------------------|----------|
| **Segment Size** | 2 million potential customers |
| **Expected Growth** | 5% annual increase |
| **Competitive Offering** | Competitors offer higher-priced eco-friendly products |

| **Positioning**  | **Data** |
|------------------|----------|
| **Product Features** | Recycled materials, Biodegradable packaging, Innovative design |
| **Presentation** | Environmental impact ads, Influencer endorsements |
| **Channels**     | Amazon, Instagram, Eco-stores |

| **Elasticity**   | **Data** |
|------------------|----------|
| **Purchase Probability** | 70% for eco-friendly product |
| **Brand Choice** | 60% prefer Brand A |
| **Purchase Quantity** | Average 2 units per transaction |

| **KYC**          | **Data** |
|------------------|----------|
| **Historic Data**| 80% online purchases |
| **Purchase Frequency** | Monthly repeat by 50% |
| **Time of Purchase** | Weekends peak sales |
| **Product Ratings** | Average 4.5 stars |

| **Marketing Mix**| **Data** |
|------------------|----------|
| **Product**      | Durable, Eco-friendly, Innovative design |
| **Price**        | Mid-range, Discounts for loyalty |
| **Promotion**    | Social media, Influencer collaborations |
| **Place**        | E-commerce, Select retail stores |

### Price Elasticity of Demand

**Definition**:
- **Price Elasticity of Demand**: Measures how the quantity demanded of a good changes in response to a change in its price.

**Formula**:
\[ E_d = \frac{\%\ \text{change in quantity demanded}}{\%\ \text{change in price}} \]

**Example Calculation**:
- Initial price of Coke: $10
- New price of Coke: $8
- Initial quantity demanded: 25 units
- New quantity demanded: 30 units

\[ \%\ \text{change in quantity demanded} = \frac{30 - 25}{25} \times 100 = 20\% \]
\[ \%\ \text{change in price} = \frac{8 - 10}{10} \times 100 = -20\% \]

\[ E_d = \frac{20\%}{-20\%} = -1 \]

**Interpretation**:
- An elasticity of -1 indicates that for every 1% decrease in price, the quantity demanded increases by 1%.

### Cross-Price Elasticity of Demand

**Definition**:
- **Cross-Price Elasticity of Demand**: Measures how the quantity demanded of one good changes in response to a change in the price of another good.

**Formula**:
\[ E_{xy} = \frac{\%\ \text{change in quantity demanded of good X}}{\%\ \text{change in price of good Y}} \]

**Example Calculation**:
- Initial price of Pepsi: $10
- New price of Pepsi: $12
- Initial quantity demanded of Coke: 30 units
- New quantity demanded of Coke: 35 units

\[ \%\ \text{change in quantity demanded of Coke} = \frac{35 - 30}{30} \times 100 = 16.67\% \]
\[ \%\ \text{change in price of Pepsi} = \frac{12 - 10}{10} \times 100 = 20\% \]

\[ E_{xy} = \frac{16.67\%}{20\%} = 0.83 \]

**Interpretation**:
- A cross-price elasticity of 0.83 indicates that a 1% increase in the price of Pepsi results in a 0.83% increase in the quantity demanded of Coke.

### Data Standardization

- **Importance**: Ensures all features are treated equally by transforming their values to fall within the same numerical range.

*Example Data Before Standardization*:
- **Age**: 20 years, 50 years, 70 years
- **Income**: $10,000, $50,000, $150,000

*Standardized Data*:
- **Age**: -1.0, 0.0, 1.0
- **Income**: -1.0, 0.0, 1.0

### Summary Statistics Using Pandas

**Pandas Describe Function**
- **Count**: Number of observations
- **Mean**: Average value
- **Std**: Standard deviation
- **Min**: Minimum value
- **25%**: 25th percentile
- **50%**: Median or 50th percentile
- **75%**: 75th percentile
- **Max**: Maximum value

*Example Output*:
```
       Age     Income
count  100.0  100.0
mean    35.0  50000.0
std     10.0  20000.0
min     18.0  10000.0
25%     25.0  30000.0
50%     35.0  50000.0
75%     45.0  70000.0
max     60.0  90000.0
```

### Correlation

- **Definition**: Measures the relationship between two variables.
- **Range**: -1 to 1
  - **-1**: Perfect negative correlation
  - **0**: No correlation
  - **1**: Perfect positive correlation

*Example*:
- Correlation between age and income:
  - **-1**: Older age, lower income
  - **0**: No relationship between age and income
  - **1**: Older age, higher income

This structured format, combined with the example data, should help in effectively presenting the customer analytics strategy.


---


### Hierarchical Clustering

**Goal:**
- Group similar observations together.
- Maximize differences between groups.

**Types of Clustering:**
1. **Hierarchical Clustering**
2. **Flat Clustering**

**Hierarchical Clustering Example:**
- **Animal Kingdom**
  - Fish
  - Cat
    - Full-developed offspring: Human, Cat, Dog
    - Immature offspring: Koalas, Kangaroos
  - Pigeon

**Detailed Example:**
- **Hierarchical Clustering of Animals:**
  - Start with individual animals (e.g., fish, cat, pigeon).
  - Merge the closest pair of animals into a single group.
  - Repeat the process until all animals are merged into a single hierarchical tree.
  - **Steps:**
    - Initially, each animal is its own cluster.
    - Merge human, cat, and dog into a "full-developed offspring" cluster.
    - Merge koalas and kangaroos into an "immature offspring" cluster.
    - Combine "full-developed offspring" and "immature offspring" under the "cat" cluster.
    - Finally, merge "fish," "cat," and "pigeon" under the "animal kingdom."

**Hierarchical Clustering Methods:**
1. **Divisive (Top-Down)**: Start with all data in one cluster and recursively split into smaller clusters.
2. **Agglomerative (Bottom-Up)**: Start with individual data points and merge them into larger clusters.

### Measuring Distance Between Observations

**Methods:**
- **Euclidean Distance:** The straight-line distance between two points in Euclidean space.
  - Example: Distance between points (2,3) and (5,7) is \(\sqrt{(5-2)^2 + (7-3)^2} = 5\).
- **Manhattan Distance:** The sum of the absolute differences of the coordinates.
  - Example: Distance between points (2,3) and (5,7) is \(|5-2| + |7-3| = 7\).
- **Maximum Distance:** The maximum of the absolute differences of the coordinates.
  - Example: Distance between points (2,3) and (5,7) is \(\max(|5-2|, |7-3|) = 4\).
- **Ward's Method:** A hierarchical clustering method that minimizes the total within-cluster variance.
  - Example: Used to measure distances in population data by minimizing the variance within each cluster.

### K-means Clustering

1. **Determine the Number of Clusters (K):**
   - Example: Suppose we want to cluster customer data into 3 clusters based on purchasing behavior.
2. **Specify Cluster Seed (Starting Centroid):**
   - Example: Initialize centroids randomly, such as (1,1), (5,7), and (9,4).
3. **Calculate the Centroid or Geometric Center:**
   - Example: After assigning each data point to the nearest centroid, recalculate the centroid as the mean of all points in the cluster.

**WCSS**: Within-Cluster Sum of Squares
- **Example:** Calculate WCSS for each cluster and sum them. Lower WCSS indicates better clustering.
  - Clusters: (1,1), (5,7), (9,4)
  - Data points assigned to clusters: [(2,2), (3,1)], [(6,8), (7,7)], [(9,3), (10,4)]
  - Centroids after recalculating: (2.5, 1.5), (6.5, 7.5), (9.5, 3.5)
  - WCSS: Sum of squared distances from each point to the centroid.

### Principal Component Analysis (PCA) + K-means Clustering

**Purpose:**
- Segmentation
- Dimensionality Reduction

**Concept:**
- Reduce dimensions from 3D to 2D using linear algebra.
- Data points lie approximately on a plane.

**Detailed Example:**
- **Data with 3 Features (X, Y, Z):**
  - Original data points: (1,2,3), (4,5,6), (7,8,9)
- **Apply PCA:**
  - Calculate the covariance matrix.
  - Determine eigenvalues and eigenvectors.
  - Project the data onto the 2D plane defined by the top 2 eigenvectors.
  - Resulting 2D data: (1.5, 2.5), (4.5, 5.5), (7.5, 8.5)
- **Apply K-means Clustering on 2D Data:**
  - Initialize centroids: (1,1), (5,5)
  - Assign points to nearest centroid.
  - Recalculate centroids: (1.5, 2.5), (7.5, 8.5)
  - Final clusters: [(1.5, 2.5)], [(7.5, 8.5)]

This detailed explanation with examples should provide a clear understanding of hierarchical clustering, measuring distances, K-means clustering, and PCA with K-means clustering.
