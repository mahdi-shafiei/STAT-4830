---
layout: course_page
title: Linear Regression - Direct Methods
---
# Linear Regression: Direct Methods

## Notebooks and Slides
- [Lecture slides](slides.pdf)
- [Colab notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/2/notebook.ipynb)

## Table of contents
1. [Introduction](#introduction)
2. [Prediction with Multiple Features](#prediction-with-multiple-features)
3. [Computing Predictions Efficiently](#computing-predictions-efficiently)
4. [Finding Optimal Weights](#finding-optimal-weights)
5. [Direct Solution Methods](#direct-solution-methods)
6. [The effect of and remedy for numerical instability](#the-effect-of-and-remedy-for-numerical-instability)
7. [QR Factorization: A More Stable Approach](#qr-factorization-a-more-stable-approach)
8. [The Limits of Direct Methods: Scaling Up](#the-limits-of-direct-methods-scaling-up)

## Introduction

Last lecture, we explored PyTorch's efficient handling of vectors and matrices. Now we'll apply these tools to a fundamental challenge in data science: prediction. Our goal is to use historical data to make accurate predictions about new situations, a problem that lies at the heart of many real-world applications.

We'll develop this idea through three key steps:
1. Converting predictions into matrix operations
2. Finding optimal predictions through optimization
3. Solving the optimization problem efficiently

The efficient matrix operations we studied last time aren't just theoretical - they're crucial for handling the large datasets we encounter in practice. As we'll see, even small improvements in computational efficiency can make the difference between a usable model and one that's too slow for real-world use.

## Prediction with Multiple Features

Consider predicting house price. A house's price typically depends on multiple characteristics: its size, age, number of bedrooms, and location. The simplest model assumes each feature contributes linearly to the price, plus some noise that captures factors our model doesn't account for:

$$ \text{price} = w_1 \cdot \text{size} + w_2 \cdot \text{age} + w_3 \cdot \text{bedrooms} + w_4 \cdot \text{location} + \text{noise} $$

In this linear relationsihp, each weight has a clear interpretation. $w_1$ represents dollars per square foot, $w_2$ tells us how price changes with age, and so on. The noise term acknowledges that real estate prices are influenced by many factors beyond our simple features.

We can write this more compactly using vector notation:
$$ y = w^T x + \epsilon $$

where $y$ is the price, $w$ contains our weights, $x$ holds the features, and $\epsilon$ represents the noise:

```python
house = {
    'size': 1500,     # x₁: sq ft
    'age': 10,        # x₂: years
    'bedrooms': 3,    # x₃: count
    'location': 0.8   # x₄: some score
}
price = 500000  # y: dollars

def predict_price(house, weights):
    """Predict house price using linear combination of features"""
    return (
        weights[0] * house['size'] +      # dollars per sq ft
        weights[1] * house['age'] +       # price change per year
        weights[2] * house['bedrooms'] +  # value per bedroom
        weights[3] * house['location']    # location premium
    )
```

Real estate data reveals why this linear approach often works well. When we plot prices against individual features, we often see roughly linear relationships with some scatter, especially within typical ranges. The scatter around these trends represents the noise in our model - those unique factors that make each house sale slightly different from what we'd predict based on features alone:

![Feature Mapping](figures/feature_mapping.png)

The challenge lies in finding the right weights while accounting for this noise. Even reasonable guesses can lead to significant errors, as we'll see by testing our model on actual house sales:

```python
# Data in PyTorch tensors
X = torch.tensor([
    [1500, 10, 3, 0.8],  # house 1
    [2100, 2,  4, 0.9],  # house 2
    [800,  50, 2, 0.3]   # house 3
], dtype=torch.float32)
y = torch.tensor([500000, 800000, 250000], dtype=torch.float32)
weights = torch.tensor([200, -1000, 50000, 100000], dtype=torch.float32)

# Make predictions
predictions = X @ weights

# Show individual errors
for i, (pred, actual) in enumerate(zip(predictions, y)):
    error = pred - actual
    print(f"House {i+1}:")
    print(f"  Predicted: ${pred:,.0f}")
    print(f"  Actual:    ${actual:,.0f}")
    print(f"  Error:     ${error:,.0f} ({error/actual:+.1%})\n")

# Compute total error
total_error = torch.sum((predictions - y)**2)
avg_error = torch.sqrt(total_error/len(y))  # Root mean square error
print(f"Total squared error: ${total_error:,.0f}")
print(f"Root mean square error: ${avg_error:,.0f}")

# Output:
# House 1:
#   Predicted: $520,000
#   Actual:    $500,000
#   Error:     $20,000 (+4.0%)
#
# House 2:
#   Predicted: $708,000
#   Actual:    $800,000
#   Error:     $-92,000 (-11.5%)
#
# House 3:
#   Predicted: $240,000
#   Actual:    $250,000
#   Error:     $-10,000 (-4.0%)
#
# Total squared error: $8,963,999,744
# Root mean square error: $54,663
```

Let's see how our predictions compare to the actual prices:

```python
plt.figure(figsize=(10, 4))
houses = range(len(y))
plt.bar(houses, y.numpy()/1000, alpha=0.5, label='Actual Price (k$)')
plt.bar(houses, predictions.numpy()/1000, alpha=0.5, label='Predicted Price (k$)')
plt.legend()
plt.title('House Prices: Predicted vs Actual')
plt.show()
```

![Predictions vs Actual](figures/predictions_vs_actual.png)

Looking at the individual errors:
```python
# Show individual errors
errors = predictions - y
for i, (pred, actual, error) in enumerate(zip(predictions, y, errors)):
    print(f"House {i+1}:")
    print(f"  Predicted: ${pred:,.0f}")
    print(f"  Actual:    ${actual:,.0f}")
    print(f"  Error:     ${error:,.0f} ({error/actual:+.1%})\n")

# Visualize percent errors
plt.figure(figsize=(10, 4))
plt.bar(houses, errors.numpy()/y.numpy() * 100)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.ylabel('Percent Error')
plt.title('Prediction Errors')
plt.show()
```

![Prediction Errors](figures/prediction_errors.png)

Our predictions aren't terrible - errors range from 4% to 12% - but can we do better? We need a systematic way to find weights that minimize our prediction error:

$$ \text{error} = \sum_{i=1}^n (y_i - w^T x_i)^2 $$

Notice what squaring does:
1. Makes all errors positive (can't have negative errors cancel out)
2. Penalizes large errors more heavily (doubling an error quadruples its contribution)
3. Creates a smooth optimization landscape we can analyze with calculus

Note: this is often called the **Root Mean Square Error (RMSE)** when we take the square root of the average.

The question is: what weights $w$ make this error as small as possible? With 4 weights and millions of houses, trying different combinations manually would be impractical. Instead, we'll:
1. Use matrix operations to compute predictions efficiently
2. Apply calculus to find the optimal weights directly
3. Solve the resulting equations using linear algebra

## Computing Predictions Efficiently

Remember from last lecture how matrix multiplication can express many computations simultaneously? This is exactly what we need here. Instead of computing each prediction in a loop, we can organize our data to use the efficient matrix operations we studied:

> **Why This Matters**
> - Matrix multiplication computes all predictions in one operation
> - Modern hardware (CPU/GPU) is optimized for matrix operations
> - The same pattern appears throughout machine learning
> - Even a small speedup (say 100ms → 10ms) becomes crucial when repeated billions of times
> - With 10,000 weight combinations × 1M houses: 10 billion computations

### From Loops to Matrices

Let's see how to convert our loop-based approach into matrix operations. Currently, we compute predictions one by one:
```python
# One-by-one approach
for house in houses:
    prediction = (w₁ * house['size'] + 
                 w₂ * house['age'] +
                 w₃ * house['bedrooms'] +
                 w₄ * house['location'])
    # Do this for EVERY house
    # Then do it ALL AGAIN for next weight combination
```

Look familiar? Each prediction is a dot product - exactly what we studied last time! We can organize all our predictions into a single matrix multiplication:

$$ \text{house}_1: [1500, 10, 3, 0.8] \cdot [w_1, w_2, w_3, w_4] = \text{prediction}_1 $$
$$ \text{house}_2: [2100, 2, 4, 0.9] \cdot [w_1, w_2, w_3, w_4] = \text{prediction}_2 $$
$$ \text{house}_3: [800, 50, 2, 0.3] \cdot [w_1, w_2, w_3, w_4] = \text{prediction}_3 $$

Stack these feature vectors into a matrix, lining up the same features in columns:

$$ X = \begin{bmatrix} 
\text{size}_1 & \text{age}_1 & \text{beds}_1 & \text{loc}_1 \\
\text{size}_2 & \text{age}_2 & \text{beds}_2 & \text{loc}_2 \\
\text{size}_3 & \text{age}_3 & \text{beds}_3 & \text{loc}_3
\end{bmatrix} = \begin{bmatrix}
1500 & 10 & 3 & 0.8 \\
2100 & 2 & 4 & 0.9 \\
800 & 50 & 2 & 0.3
\end{bmatrix} $$

> **Quick Exercise**
> Before running the code below:
> 1. What will be the prediction for house 1? (Hint: multiply first row by weights)
> 2. Which house do you think will have the highest predicted price? Why?

Now one matrix multiplication computes ALL predictions:
```python
# Matrix approach
X = torch.tensor([
    [1500, 10, 3, 0.8],  # All house 1 features
    [2100, 2,  4, 0.9],  # All house 2 features
    [800,  50, 2, 0.3]   # All house 3 features
])
w = torch.tensor([200, -1000, 50000, 100000])
predictions = X @ w  # All predictions in one operation
```

> **From Last Lecture**
> Remember how PyTorch optimizes matrix operations:
> - Uses CPU's SIMD instructions (compute multiple values at once)
> - Accesses memory in cache-friendly patterns
> - Leverages optimized BLAS libraries

Let's measure the impact:
```
Dataset Size | Loop Time  | Matrix Time | Speedup
------------------------------------------------
     1,000   |   0.21ms  |    0.01ms   |   21x
    10,000   |   1.79ms  |    0.05ms   |   34x
   100,000   |  19.39ms  |    0.58ms   |   33x
 1,000,000   | 196.33ms  |    5.43ms   |   36x
```

This speedup is crucial because in later lectures, we'll need to compute predictions repeatedly when using iterative optimization methods. But for now, we'll discover something remarkable:

Our optimization problem (finding weights that minimize error) can be transformed into a problem we already know how to solve - a system of linear equations! Instead of trying different weights or using calculus, we can:
1. Convert "minimize the error" into a special set of equations
2. Solve these equations using techniques from linear algebra
3. Find the optimal weights directly

In the next section, we'll see how this transformation works and why the resulting equations give us the best possible weights. The efficiency of matrix operations will make solving these equations practical even for large datasets.

## Finding Optimal Weights

Remember our 10% error on house prices? Let's discover why calculus and linear algebra together give us a direct path to the best weights. Let's start simple: imagine we only have sizes and prices for two houses:
- House 1: 1000 sq ft → 300k dollars
- House 2: 2000 sq ft → 600k dollars

Notice that when size doubles (1000 → 2000), price doubles too (300k → 600k). This suggests a perfect linear relationship!

Our error for a given weight $w$ (price per sq ft, in $k$) is:

$$ \text{error}(w) = (300 - 1000w)^2 + (600 - 2000w)^2 $$

To minimize this error, we use calculus: set the derivative to zero and solve. Taking the derivative (chain rule gives us -2 times each feature) and setting to zero:

$$ -2(1000)(300 - 1000w) - 2(2000)(600 - 2000w) = 0 $$

Collecting terms, we get:

$$ (1000^2 + 2000^2)w = 1000(300) + 2000(600) $$

This solves our simple case, but with more features, the calculus gets messy. Here's where matrix notation helps. If we write our data as matrices:

$$ X = \begin{bmatrix} 1000 \\ 2000 \end{bmatrix}, \quad y = \begin{bmatrix} 300 \\ 600 \end{bmatrix} $$

Then our equation becomes:

$$ (X^TX)w = X^Ty $$

With our numbers, that's:

$$ 5,000,000w = 1,500,000 $$

Solving gives $w = 0.3$ - or 300 dollars per square foot. This confirms what we saw in the data: since price per square foot is constant at 300 dollars, doubling square footage (1000 → 2000) exactly doubles the price (300k → 600k).

This pattern extends to multiple features in a natural way. Just as we took the derivative with respect to one weight, with multiple weights we take partial derivatives with respect to each weight - something you've likely seen before in multivariable calculus or machine learning courses. Setting these partial derivatives to zero (the gradient $\nabla_w\text{error} = 0$) gives us:
1. One equation per weight
2. Each equation is linear in the weights
3. All equations involve the same features and prices

When we organize these equations using matrix notation:

$$ (X^TX)w = X^Ty $$

This system, known as the **normal equations**, is one we can solve directly! The system is called the normal equations because the error vector $(Xw - y)$ becomes orthogonal (normal) to the column space of $X$ at the solution. They give us a direct path to the optimal weights through linear algebra, as discuss in the next section.

## Direct Solution Methods

Now we'll discuss three methods for solving the normal equations: Gaussian elimination, LU factorization, and QR factorization. We'll see how these methods scale with the size of our dataset and how they compare in terms of computational efficiency and numerical stability.

Let's first look at our house price example with three features:
```python
X = torch.tensor([
    [1500, 10, 3],    # house 1: size, age, bedrooms
    [2100, 2,  4],    # house 2
    [800,  50, 2],    # house 3
    [1800, 15, 3]     # house 4
], dtype=torch.float32)
y = torch.tensor([500000, 800000, 250000, 550000], dtype=torch.float32)
```

The normal equations $(X^TX)w = X^Ty$ give us a system $Aw = b$ where:
- $A = X^TX$ is a square matrix with size (number of features × number of features)
- $b = X^Ty$ combines features and prices through dot products

Note that even with four houses (or four thousand!), our system is just $3 \times 3$ because we have 3 features! The matrix $X$ is $4 \times 3$, but $X^TX$ is always $p \times p$, matching the number of weights we need to find.

When we multiply $X^TX$, each entry combines feature vectors across all houses:

$$ A = X^TX = \begin{bmatrix} 
\mathbf{size} \cdot \mathbf{size} & \mathbf{size} \cdot \mathbf{age} & \mathbf{size} \cdot \mathbf{beds} \\
\mathbf{age} \cdot \mathbf{size} & \mathbf{age} \cdot \mathbf{age} & \mathbf{age} \cdot \mathbf{beds} \\
\mathbf{beds} \cdot \mathbf{size} & \mathbf{beds} \cdot \mathbf{age} & \mathbf{beds} \cdot \mathbf{beds}
\end{bmatrix} $$

where each $\mathbf{size}$, $\mathbf{age}$, and $\mathbf{beds}$ is a vector containing that feature for all houses.

For example:
- $(X^TX)_{11} = \mathbf{size} \cdot \mathbf{size} = 1500^2 + 2100^2 + 800^2 + 1800^2$
- $(X^TX)_{12} = \mathbf{size} \cdot \mathbf{age} = 1500(10) + 2100(2) + 800(50) + 1800(15)$

This structure explains why:
- Diagonal entries sum squares of each feature (always positive, often large)
- Off-diagonal entries show how pairs of features vary together

The vector $b = X^Ty$ similarly combines features with prices through dot products.

The resulting system $Aw = b$ looks like:

$$ \begin{bmatrix} 
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
\begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix} =
\begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix} $$

### Solving the System: Back to Linear Algebra

We've seen systems like this before in linear algebra. There are several ways to solve such systems. We'll consider three so-called *direct methods* in this lecture: Gaussian elimination, LU factorization, and QR factorization. But how do we choose between them?

Two critical factors determine which method to use:
1. Computational Efficiency
   - Measured by number of arithmetic operations
   - Becomes critical with large systems
   - Affects running time directly
2. Numerical Stability
   - Determines how measurement errors get amplified
   - Critical when features are correlated
   - Can make fast methods unreliable

The tension between these factors - and why practice often favors stability over raw speed - will become clear as we explore each method.

To understand these methods and their trade-offs, let's:
1. Work through each method on our 3×3 system
2. See how both cost and stability scale with larger matrices
3. Discover why forming $X^TX$ (while tempting) can be lead to poor numerical stability

Let's start with Gaussian elimination:

#### Gaussian Elimination

Gaussian elimination solves equations by systematically removing variables. The idea is simple: use one equation to eliminate a variable from the others, then repeat. We'll create zeros below the diagonal one column at a time, turning our system into an equivalent triangular form that's easy to solve by back-substitution.

*Goal*: Create zeros below the diagonal systematically

#### Step 1: First Elimination
*Goal*: Create zeros in first column below $a_{11}$

*Compute multipliers*:

$$ m_{21} = \displaystyle\frac{a_{21}}{a_{11}} \quad \text{and} \quad m_{31} = \displaystyle\frac{a_{31}}{a_{11}} $$

*After row operations*:

$$ \begin{array}{c|c}
\begin{matrix} 
a_{11} & a_{12} & a_{13} \\[0.7em]
0 & a_{22}' & a_{23}' \\[0.7em]
0 & a_{32}' & a_{33}'
\end{matrix} &
\begin{matrix}
b_1 \\[0.7em]
b_2' \\[0.7em]
b_3'
\end{matrix}
\end{array} \qquad \text{(24 operations: 12 multiplications, 12 subtractions)} $$

where $a_{22}' = a_{22} - m_{21}a_{12}$ and similarly for other entries.

#### Step 2: Second Elimination
*Goal*: Create zero in second column below $a_{22}'$

*Compute multiplier*:

$$ m_{32} = \displaystyle\frac{a_{32}'}{a_{22}'} $$

*After row operations*:

$$ \begin{array}{c|c}
\begin{matrix} 
a_{11} & a_{12} & a_{13} \\[0.7em]
0 & a_{22}' & a_{23}' \\[0.7em]
0 & 0 & a_{33}''
\end{matrix} &
\begin{matrix}
b_1 \\[0.7em]
b_2' \\[0.7em]
b_3''
\end{matrix}
\end{array} \qquad \text{(8 operations: 4 multiplications, 4 subtractions)} $$

#### Step 3: Back-substitution
*Solve from bottom to top*:

$$ \begin{aligned}
w_3 &= \displaystyle\frac{b_3''}{a_{33}''} && \text{(1 division)} \\[0.7em]
w_2 &= \displaystyle\frac{b_2' - a_{23}'w_3}{a_{22}'} && \text{(2 ops + 1 division)} \\[0.7em]
w_1 &= \displaystyle\frac{b_1 - a_{12}w_2 - a_{13}w_3}{a_{11}} && \text{(4 ops + 1 division)}
\end{aligned} $$

For our 3×3 system, we needed 6 divisions, 19 multiplications, and 19 additions or subtractions. Looking at how these counts arise reveals the pattern: each elimination step processes one column, requiring operations proportional to the size of the remaining matrix. For an n×n system, this pattern leads to approximately $\frac{2n^3}{3}$ operations for elimination and another $\frac{n^2}{2}$ for back-substitution.

To find optimal weights, we need two steps:
1. Form the normal equations by computing $X^TX$ and $X^Ty$
2. Solve the resulting system using Gaussian elimination

Both steps can be expensive! With $n$ houses and $p$ features:
- Computing $X^TX$ requires $np^2$ operations (each entry needs $n$ multiplications)
- Computing $X^Ty$ requires $np$ operations
- Solving the $p \times p$ system needs $\frac{2p^3}{3}$ operations

Which step dominates depends on our problem:
- More houses ($n$ large): Formation cost $np^2$ grows
- More features ($p$ large): Solution cost $\frac{2p^3}{3}$ grows faster

For example:
- $1000$ houses, $10$ features:
  * Formation: $1000 \times 10^2 = 100,000$ operations
  * Solution: $\frac{2(10)^3}{3} \approx 667$ operations
  * Formation dominates!

- $1000$ houses, $100$ features:
  * Formation: $1000 \times 100^2 = 10$ million operations
  * Solution: $\frac{2(100)^3}{3} \approx 667,000$ operations
  * Both costs matter

In modern statistics, we often have both:
- Many houses ($n$ large): more data helps prediction
- Many features ($p$ large): interaction terms, location indicators, seasonal effects

This makes both formation and solution costs important to consider!

Next, we'll explore $LU$ factorization - a clever way to reorganize Gaussian elimination that becomes especially valuable when we need to update our predictions with new house prices. Instead of solving the entire system again, we'll see how to reuse much of our previous work.

### LU Factorization


Imagine this scenario:
- You've just computed optimal weights for 1000 houses
- Then 100 new houses sell, with different prices
- Market conditions shift existing home values
- Seasonal patterns affect current listings
 
Each change means new optimal weights. Can we avoid redoing all our work?

#### From Elimination to Factorization: A Key Insight

Remember our elimination steps? Let's look closer at what we're doing:

*Step 1: Create zeros in first column*

Start with:
$$ \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\[0.7em]
\times & a_{22} & a_{23} \\[0.7em]
\times & a_{32} & a_{33}
\end{bmatrix} $$

Compute multipliers to eliminate marked entries:
$$ m_{21} = \displaystyle\frac{a_{21}}{a_{11}} \quad \text{and} \quad m_{31} = \displaystyle\frac{a_{31}}{a_{11}} $$

When we subtract $m_{21}$ times row 1 from row 2:
- The multiplier $m_{21}$ goes into L (recording what we did)
- A zero appears in U (showing what we achieved)

After both eliminations:
$$ \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\[0.7em]
& a_{22}' & a_{23}' \\[0.7em]
& a_{32}' & a_{33}'
\end{bmatrix} = 
\underbrace{\begin{bmatrix}
1 & & \\[0.7em]
m_{21} & 1 & \\[0.7em]
m_{31} & \text{next} & 1
\end{bmatrix}}_{\text{L (multipliers used)}} \times
\underbrace{\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\[0.7em]
& a_{22}' & a_{23}' \\[0.7em]
& a_{32}' & a_{33}'
\end{bmatrix}}_{\text{U (current state)}} $$

The "next" entry in L will come from our next elimination step. Each column of L is completed as we eliminate variables in the corresponding column of U.

*Step 2: Create zero in second column*

Compute the next multiplier:
$$ m_{32} = \displaystyle\frac{a_{32}'}{a_{22}'} $$

This gives our final factorization:
$$ A = \underbrace{\begin{bmatrix}
1 & & \\[0.7em]
m_{21} & 1 & \\[0.7em]
m_{31} & m_{32} & 1
\end{bmatrix}}_{\text{L (elimination history)}} \times
\underbrace{\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\[0.7em]
& a_{22}' & a_{23}' \\[0.7em]
& & a_{33}''
\end{bmatrix}}_{\text{U (eliminated system)}} $$

Why does this work?
Each elimination step:
1. Computes a multiplier $m$
2. Creates a zero in $U$
3. Records $m$ in $L$
 
$L$ captures HOW we eliminated (our steps). $U$ shows WHAT we achieved (zeros below diagonal)

Why is this pattern useful? Because it splits our solution process into two parts:
1. The elimination recipe ($L$): records exactly how we created each zero
2. The eliminated system ($U$): shows the result of following that recipe

Once we have this split, solving $Aw = b$ becomes a two-step process:
1. Use $L$ to solve $Ly = b$ (forward substitution)
2. Use $U$ to solve $Uw = y$ (back substitution)

Best of all, when house prices change and we get a new $b$, we can reuse our factorization! The elimination recipe ($L$) and eliminated system ($U$) stay the same - we just need to follow the recipe with new prices.

#### Solving with $L$ and $U$

Instead of solving $Aw = b$ directly, we solve two simpler triangular systems:

*Step 1: Forward substitution ($Ly = b$)*
$$ \begin{aligned}
y_1 &= b_1 && \text{(1's on diagonal)} \\[0.7em]
y_2 &= b_2 - m_{21}y_1 && \text{(2 operations)} \\[0.7em]
y_3 &= b_3 - m_{31}y_1 - m_{32}y_2 && \text{(4 operations)}
\end{aligned} $$

*Step 2: Back substitution ($Uw = y$)*
$$ \begin{aligned}
w_3 &= \displaystyle\frac{y_3}{u_{33}} && \text{(1 division)} \\[0.7em]
w_2 &= \displaystyle\frac{y_2 - u_{23}w_3}{u_{22}} && \text{(2 ops + 1 division)} \\[0.7em]
w_1 &= \displaystyle\frac{y_1 - u_{12}w_2 - u_{13}w_3}{u_{11}} && \text{(4 ops + 1 division)}
\end{aligned} $$

#### Why This Helps: A Practical Example

When house prices change:
- Matrix $A = X^TX$ stays the same (features haven't changed)
- Only vector $b = X^Ty$ changes (new prices affect $y$)

Let's see the dramatic impact on computation:

| Operation | First Time | Each Update |
|-----------|------------|-------------|
| Factor A = LU | 667,000 | (already done!) |
| Solve Ly = b | 5,000 | 5,000 |
| Solve Uw = y | 5,000 | 5,000 |
| **Total** | 677,000 | 10,000 |

With daily updates over a year:
- Without $LU$: $247$ million operations
- With $LU$: $4.3$ million operations
- **$98\%$ reduction in computation!**

This dramatic speedup shows why factorization is useful: by separating HOW we eliminate ($L$) from WHAT we achieve ($U$), we can reuse our work when only prices change. Let's see this in action with PyTorch:

```python
# Form A = X^TX for our house price example
X = torch.tensor([
    [1500., 10., 3.],    # house 1: size, age, bedrooms
    [2100., 2., 4.],     # house 2
    [800., 50., 2.],     # house 3
    [1800., 15., 3.]     # house 4
])
A = X.T @ X

# Compute LU factorization
L, U = torch.lu(A)
print("L =\n", L)
print("\nU =\n", U)

# Output:
# L = tensor([[1.0000, 0.0000, 0.0000],
#            [0.3214, 1.0000, 0.0000],
#            [0.0018, 0.0842, 1.0000]])
#
# U = tensor([[1.1225e+07, 2.7400e+04, 4.8000e+03],
#            [0.0000e+00, 2.8934e+03, 1.2000e+01],
#            [0.0000e+00, 0.0000e+00, 2.0000e+00]])
```

Before moving on, it's worth noting that when A has the special form X^TX, there's another factorization called Cholesky that's twice as fast as LU. While we won't explore it here, keep it in mind for future reference.

## The effect of and remedy for numerical instability

The LU approach has a hidden weakness that becomes clear when we think geometrically about our features. When we plot two features against each other, we can see two very different situations:

![Condition Number Visualization](figures/condition_number_viz.png)

- Left: Independent features form a "round" cloud - changes in one feature don't tell us much about the other
- Right: Nearly dependent features form a "skinny" cloud - knowing one feature almost completely determines the other

This "skinniness" makes our problem ill-conditioned - small changes in the data can cause large changes in our solution. The condition number κ(X) measures exactly this:
- It's the ratio of the largest to smallest "stretching" our data does in any direction
- A condition number of 100 means the longest direction is 100 times the shortest
- The larger this ratio, the more sensitive our solution becomes to small errors

More precisely, the condition number κ(X) is computed as the ratio of the largest to smallest singular values of X (κ(X) = σ_max/σ_min). These singular values come from the SVD decomposition we mentioned at the end of the last lecture - they exactly measure the amount of stretching X does in each direction. We'll explore this connection in detail in later lectures, but the interested reader can consult the [mathematical foundations of SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) for a deeper understanding.

For example, in our visualization:
- Independent features have κ(X) ≈ 1.0 (nearly circular)
- Nearly dependent features have κ(X) ≈ 201.2 (very elongated)

Now comes the crucial issue: when we form X^TX in the normal equations, we're squaring this condition number:
- Independent features: κ(X^TX) ≈ 1.1 (stays well-conditioned)
- Nearly dependent features: κ(X^TX) ≈ 40,580.2 (becomes extremely ill-conditioned)

This squaring effect explains why small measurement errors can lead to large errors in our weights. Let's see what happens when we add a tiny perturbation to our data:

```python
# Add a small perturbation (0.0001% noise)
X_perturbed = X * (1 + torch.randn(*X.shape) * 1e-6)

# Solve both systems using normal equations
def solve_normal_equations(X, y):
    XtX = X.T @ X
    Xty = X.T @ y
    return torch.linalg.solve(XtX, Xty)

# Create target that depends only on x1
y = x1 + torch.randn(n) * 0.1

w1 = solve_normal_equations(X, y)
w2 = solve_normal_equations(X_perturbed, y)

print("\nWeights with original data:", w1)
print("Weights with perturbed data:", w2)
print("\nRelative change in weights:", torch.abs((w2 - w1)/w1))
print("But predictions change very little:", 
      torch.norm(X@w2 - X@w1)/torch.norm(X@w1))
```

The results are striking:
- Original weights: [1.1, -0.1]
- Perturbed weights: [1.8, -0.8]
- Individual weights change by ~80%
- But predictions barely change at all (<0.1%)

This reveals the core problem: when features are nearly dependent (creating a large condition number):
1. Many different weight combinations give similar predictions
2. Tiny data changes can cause large swings between these combinations
3. Forming X^TX makes this instability much worse by squaring the condition number

## QR Factorization: A More Stable Approach

The key to avoiding this numerical instability is to never form X^TX in the first place. QR factorization offers exactly this solution by working directly with X. Instead of squaring the condition number through X^TX, it transforms our features into an orthogonal basis that preserves their numerical properties.

The QR approach splits X into two matrices:
- Q: Has orthonormal columns (perpendicular and unit length)
- R: Upper triangular, captures feature relationships

For a data matrix X with n rows and p columns (an n×p matrix):
- Q is n×n 
- R is n×p (same shape as X)
- Only the top p×p part of R is needed for solving 

For example, with p=3 features, R has this structure:

$$ R = \begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
0 & r_{22} & r_{23} \\
0 & 0 & r_{33} \\
\hline
0 & 0 & 0 \\
\vdots & \vdots & \vdots \\
0 & 0 & 0
\end{bmatrix} \begin{array}{l}
\leftarrow \text{upper triangular } p \times p \text{ part} \\
\\
\leftarrow \text{zeros in remaining rows}
\end{array} $$

Instead of eliminating variables like LU, QR factorization transforms our features to make them independent of each other.

You've likely seen this idea before in linear algebra - it's the same principle behind the Gram-Schmidt process, which turns any set of vectors into orthogonal ones. QR factorization is essentially an efficient, numerically stable way to do Gram-Schmidt. While we won't dive into the details of how it's computed (that's a topic for numerical linear algebra courses), here's what it does:

1. Q is like a rotation matrix - it turns our correlated features into perpendicular ones
2. R records how to combine these new perpendicular features to get back our original ones
3. Together they give us X = QR, but with Q's columns being perfectly perpendicular

Think of it like untangling a bunch of twisted ropes (our correlated features) into neat parallel lines (orthogonal features). The original ropes might be all tangled together, making it hard to see their individual effects. But after "untangling" them with Q, we can see exactly how each one contributes.

Let's see this in action with our house price example:
```python
X = torch.tensor([
    [1500., 10., 3.],    # house 1: size, age, bedrooms
    [2100., 2., 4.],     # house 2
    [800., 50., 2.],     # house 3
    [1800., 15., 3.]     # house 4
])

# Compute QR factorization
Q, R = torch.qr(X)

# Check orthogonality: Q^T @ Q should be identity
print("Q^T @ Q =\n", Q.T @ Q)

# Check that Q preserves lengths
original_lengths = torch.norm(X, dim=0)
q_lengths = torch.norm(Q, dim=0)
print("\nOriginal feature lengths:", original_lengths)
print("Q column lengths:", q_lengths)

# Output:
# Q^T @ Q = 
# tensor([[1.0000, 0.0000, 0.0000],
#         [0.0000, 1.0000, 0.0000],
#         [0.0000, 0.0000, 1.0000]])
```

Notice:
- Q's columns are orthogonal (perpendicular) and normalized (length 1)
- R captures how to combine these new directions to get our original ones
- Together they give us X = QR, but with Q's columns being perfectly perpendicular

### Solving with QR

To find optimal weights, we start with our original problem:

$$ Xw = y $$

Since $X = QR$, this becomes:

$$ QRw = y $$

Now comes the key insight: since Q's columns are perpendicular (orthogonal), multiplying both sides by Q^T is like "untangling" our equations. Why? Because Q^TQ = I, meaning all the cross-terms cancel out perfectly:

$$ \begin{aligned}
Q^T(QRw) &= Q^Ty \\
(Q^TQ)Rw &= Q^Ty \\
IRw &= Q^Ty \\
Rw &= Q^Ty
\end{aligned} $$

This is beautiful! We've turned our problem into a triangular system ($Rw = Q^Ty$) without ever forming X^TX. More precisely, 
- If our data matrix $X$ is 4×3 (4 houses × 3 features)
- $Q$ is 4×4 (as many rows as we have houses)
- $R$ is 4×3 (same shape as X)
- Only the top 3×3 part of $R$ (matching our number of features) is needed for solving

Since we're solving for 3 weights (one per feature), we only need 3 equations. The top 3×3 part of R gives us exactly these equations in upper triangular form. This is why we can focus on:

$$ R = \begin{bmatrix}
r_{11} & r_{12} & r_{13} \\
0 & r_{22} & r_{23} \\
0 & 0 & r_{33}
\end{bmatrix}, \quad
Rw = c \quad \text{where } c = Q^Ty $$

Just as with LU factorization, we can solve this triangular system efficiently using back substitution. Starting from the bottom row (which has only one unknown) and working up:

$$ \begin{aligned}
w_3 &= c_3/r_{33} \\
w_2 &= (c_2 - r_{23}w_3)/r_{22} \\
w_1 &= (c_1 - r_{12}w_2 - r_{13}w_3)/r_{11}
\end{aligned} $$

### Comparing LU and QR

The key difference between QR and LU is that we get an triangular system directly from QR, without ever forming the numerically unstable X^TX. This process is both numerically stable (avoiding condition number squaring) and computationally efficient (requiring just one triangular solve). Let's compare both approaches:

$$ \begin{aligned}
&\text{LU approach: } np^2 \text{ to form } X^TX \text{, then } \frac{2p^3}{3} \text{ to factor it} \\
&\text{QR approach: } 2np^2 \text{ to factor X directly}
\end{aligned} $$

When we have many more houses than features ($n \gg p$), the $np^2$ terms dominate. In this case, LU should be about twice as fast as QR since it needs $np^2$ operations compared to QR's $2np^2$. However, as we'll see, this theoretical speed advantage often isn't worth the potential loss in numerical accuracy.

Let's verify QR's numerical stability with a comprehensive test:

Let's design an experiment that mirrors real-world challenges in house price prediction. We'll start by generating synthetic data that captures natural relationships - like how larger houses tend to have more bedrooms - and then introduce a deliberately challenging feature to test how our methods handle numerical stress.

```python
# Generate synthetic house data
torch.manual_seed(42)
n_samples = 10000

# Create base features with realistic scales
sqft = torch.rand(n_samples) * 2000 + 1000  # Uniform between 1000-3000
age = torch.rand(n_samples) * 30            # Uniform between 0-30
# Make bedrooms correlate naturally with size
bedrooms = torch.clamp(sqft/1000 + torch.randn(n_samples) * 0.5, 1, 5)

# Stack features and add known weights
X = torch.stack([sqft, age, bedrooms], dim=1)
true_weights = torch.tensor([200.0, -1000.0, 50000.0])  # $/sqft, $/year, $/bedroom
y = X @ true_weights + torch.randn(n_samples) * 100     # Add small noise

# Add highly correlated feature to test stability
X = torch.cat([X, X[:, 0:1] + torch.randn(n_samples, 1) * 10], dim=1)

# Compare condition numbers
print(f"Condition number of X: {torch.linalg.cond(X):.1f}")
print(f"Condition number of X^TX: {torch.linalg.cond(X.T @ X):.1f}")

# Solve using both methods
XtX = X.T @ X
Xty = X.T @ y
LU, pivots = torch.linalg.lu_factor(XtX)
w_lu = torch.linalg.lu_solve(LU, pivots, Xty.unsqueeze(1)).squeeze(1)

Q, R = torch.linalg.qr(X)
w_qr = torch.linalg.solve_triangular(R, Q.T @ y.unsqueeze(1), upper=True).squeeze(1)

# Compare results
print("\nWeight estimates (first 3 features):")
print(f"True weights: {true_weights}")
print(f"LU weights:   {w_lu[:3]}")
print(f"QR weights:   {w_qr[:3]}")

# Compare prediction errors
rmse_lu = torch.sqrt(torch.mean((X @ w_lu - y)**2))
rmse_qr = torch.sqrt(torch.mean((X @ w_qr - y)**2))
print(f"\nRMSE:")
print(f"LU: {rmse_lu:.2f} dollars")
print(f"QR: {rmse_qr:.2f} dollars")
```

The output shows:
```
Condition number of X: 6261.7
Condition number of X^TX: 39208800.0

Weight estimates (first 3 features):
True weights: tensor([  200., -1000., 50000.])
LU weights:   tensor([  209.2539, -1000.0673, 50001.0234])
QR weights:   tensor([  199.9693, -1000.0654, 50000.3867])

RMSE:
LU: 138.04 dollars
QR: 101.08 dollars
```

This experiment demonstrates key numerical effects in least squares problems. Forming X^TX squares the condition number from 6,262 to 39.2 million - a worst-case scenario with highly correlated features. This affects the accuracy of weights computed through normal equations with LU factorization, visible in the price per square foot coefficient (209.25 versus true 200). QR factorization, by avoiding explicit formation of X^TX, maintains better accuracy (199.97) and achieves lower prediction error (RMSE 101.08 dollars versus 138.04).

The choice between methods depends on the problem structure. Both approaches are backward stable in their respective formulations - QR for the original least squares problem, and Cholesky-based normal equations for the X^TX formulation. For well-conditioned problems (κ(X) close to 1), normal equations with Cholesky factorization can be faster and perfectly adequate. QR factorization, while requiring roughly twice the operations (2np² versus np²), provides better numerical stability by inheriting X's condition number directly rather than squaring it. In practice, implementations use "thin QR" where Q is n×p instead of n×n (since we only need p orthogonal vectors), making it more efficient. Neither method completely solves ill-conditioning, but QR handles it more gracefully by avoiding the explicit formation of X^TX.

## The Limits of Direct Methods: Scaling Up

Direct methods face a hard constraint: they must complete their entire computation before producing any solution. For problems with millions of observations, this means waiting minutes for an answer. In massive-scale applications like recommender systems, this delay becomes impractical.

This constraint motivates iterative methods. Instead of computing an exact solution upfront, they produce increasingly accurate predictions over time. This trade-off - accepting approximate answers for faster results - often matters more than theoretical perfection.

Direct methods remain competitive for moderate-sized problems, especially those with special structure like sparsity. But when datasets grow large or quick answers matter more than perfect ones, iterative methods become essential. We'll explore these methods next.

