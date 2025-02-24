---
layout: course_page
title: Stochastic Gradient Descent - Insights from a Noisy Quadratic Model
---

# Stochastic Gradient Descent: Insights from a Noisy Quadratic Model

## Table of contents
1. [Introduction](#introduction)
2. [Problem setup: Noisy quadratic model](#problem-setup-noisy-quadratic-model)
3. [Minibatch SGD as gradient descent with noise](#minibatch-sgd-as-gradient-descent-with-noise)
4. [Higher-dimensional challenges](#higher-dimensional-challenges)
5. [Momentum](#momentum)
6. [Exponential moving averages](#exponential-moving-averages)
7. [Preconditioning](#preconditioning)
8. [Experimental comparisons](#experimental-comparisons)
9. [Complete code](#complete-code)

## Introduction

In our previous lecture, we examined stochastic gradient descent (SGD) in the context of large-scale optimization problems. We saw that SGD offers a memory-efficient approach for problems with massive datasets by sampling random data points rather than processing the entire dataset at once. We analyzed its convergence properties, both in expectation and variance, and observed the trade-offs involved in choosing the step size.

Today, we'll extend this understanding by studying SGD through the lens of a noisy quadratic model. This simple but powerful framework will give us valuable insights into practical modifications of SGD used in deep learning:

1. **Momentum**: How carrying velocity from previous updates can improve convergence
2. **Exponential moving averages**: How averaging parameter values can reduce variance
3. **Higher-dimensional challenges**: What happens when we optimize multiple parameters simultaneously
4. **Preconditioning**: How reshaping the optimization landscape can accelerate convergence

These modifications are crucial for training deep neural networks efficiently. While the basic SGD algorithm works well for simpler problems, these extensions help navigate the complex, high-dimensional loss landscapes of modern deep learning models.

Our goal today is to give you intuition for why these modifications work, based on a theoretical but accessible model. We'll follow the excellent exposition in a 2019 paper by researchers at Google, DeepMind, University of Toronto, and Anthropic.

## Problem setup: Noisy quadratic model

Let's begin by setting up our model. We'll focus on a simple multivariate quadratic problem in two dimensions. While this might seem overly simplistic, the insights we gain will generalize to higher dimensions and more complex loss functions.

Fix constants $h_1 > h_2 > 0$ and $\sigma_1, \sigma_2 > 0$ (with no ordering on the noise terms yet). We consider a random variable $x \in \mathbb{R}^2$, satisfying:

- Mean zero: $\mathbb{E}[x_1] = \mathbb{E}[x_2] = 0$
- Independent components: $\mathbb{E}[x_1x_2] = \mathbb{E}[x_1]\mathbb{E}[x_2] = 0$
- Variance: Each component has variance $\sigma_i^2$, meaning $\mathbb{E}[x_i^2] = \sigma_i^2$ for $i = 1, 2$

Our model is a simple least squares problem:

$$
L(w) = \frac{1}{2}\mathbb{E}_{(x_1, x_2) \sim P}\left[h_1(x_1 - w_1)^2 + h_2(x_2 - w_2)^2\right]
$$

This looks similar to the model we worked with in lecture 6, but now we have two parameters $w_1$ and $w_2$ instead of just one. We can define the sample-wise loss:

$$
\ell(w; (x_1, x_2)) = \frac{1}{2}\left(h_1(x_1 - w_1)^2 + h_2(x_2 - w_2)^2\right)
$$

By definition:
$$
L(w) = \mathbb{E}_{(x_1, x_2) \sim P}[\ell(w; (x_1, x_2))]
$$

### Expanding the population loss

Let's expand $L(w)$ to get a more explicit form:

$$
\begin{aligned}
L(w) &= \frac{1}{2}\mathbb{E}\left[h_1(x_1 - w_1)^2 + h_2(x_2 - w_2)^2\right] \\
&= \frac{1}{2}\mathbb{E}\left[h_1(x_1^2 - 2x_1w_1 + w_1^2) + h_2(x_2^2 - 2x_2w_2 + w_2^2)\right] \\
&= \frac{1}{2}\left(h_1\mathbb{E}[x_1^2] - 2h_1\mathbb{E}[x_1]w_1 + h_1w_1^2 + h_2\mathbb{E}[x_2^2] - 2h_2\mathbb{E}[x_2]w_2 + h_2w_2^2\right)
\end{aligned}
$$

Since $\mathbb{E}[x_1] = \mathbb{E}[x_2] = 0$ and $\mathbb{E}[x_1^2] = \sigma_1^2$, $\mathbb{E}[x_2^2] = \sigma_2^2$, we get:

$$
\begin{aligned}
L(w) &= \frac{1}{2}\left(h_1\sigma_1^2 + h_1w_1^2 + h_2\sigma_2^2 + h_2w_2^2\right) \\
&= \frac{1}{2}\left(h_1w_1^2 + h_2w_2^2\right) + \frac{1}{2}\left(h_1\sigma_1^2 + h_2\sigma_2^2\right)
\end{aligned}
$$

So our population loss is:

$$\boxed{L(w) = \frac{1}{2}\left(h_1w_1^2 + h_2w_2^2\right) + \frac{1}{2}\left(h_1\sigma_1^2 + h_2\sigma_2^2\right)}$$

The second term is constant with respect to $w$, so the minimizer is clearly $w^* = (0, 0)$.

### Gradient and noise calculation

Now, let's look at the gradient of $L(w)$ and how it relates to the gradient of the sample loss $\ell(w; (x_1, x_2))$.

The gradient of the population loss is:
$$\nabla L(w) = \begin{pmatrix} h_1w_1 \\ h_2w_2 \end{pmatrix}$$

For the sample loss, we have:
$$\nabla \ell(w; (x_1, x_2)) = \begin{pmatrix} h_1(w_1 - x_1) \\ h_2(w_2 - x_2) \end{pmatrix}$$

Now, let's compute the expectation of the gradient of the sample loss:

$$
\begin{aligned}
\mathbb{E}[\nabla \ell(w; (x_1, x_2))] &= \mathbb{E}\begin{pmatrix} h_1(w_1 - x_1) \\ h_2(w_2 - x_2) \end{pmatrix} \\
&= \begin{pmatrix} h_1w_1 - h_1\mathbb{E}[x_1] \\ h_2w_2 - h_2\mathbb{E}[x_2] \end{pmatrix} \\
&= \begin{pmatrix} h_1w_1 \\ h_2w_2 \end{pmatrix} \\
&= \nabla L(w)
\end{aligned}
$$

This confirms an important result: the expected gradient of the sample loss equals the gradient of the population loss. Mathematically:

$$\boxed{\nabla L(w) = \mathbb{E}[\nabla \ell(w; (x_1, x_2))]}$$

This property, known as differentiating under the integral sign, is what makes stochastic gradient descent work. It means that the stochastic gradient is an unbiased estimator of the true gradient.

Let's denote the noise in the stochastic gradient as $\varepsilon$. We can write:

$$\nabla \ell(w; (x_1, x_2)) = \nabla L(w) + \varepsilon$$

where:
$$\varepsilon = \begin{pmatrix} -h_1x_1 \\ -h_2x_2 \end{pmatrix}$$

The covariance matrix of this noise is:

$$
\begin{aligned}
\text{Cov}(\varepsilon) &= \mathbb{E}[\varepsilon\varepsilon^T] - \mathbb{E}[\varepsilon]\mathbb{E}[\varepsilon]^T \\
&= \mathbb{E}\begin{pmatrix} h_1^2x_1^2 & h_1h_2x_1x_2 \\ h_1h_2x_1x_2 & h_2^2x_2^2 \end{pmatrix} - \begin{pmatrix} 0 \\ 0 \end{pmatrix}\begin{pmatrix} 0 & 0 \end{pmatrix} \\
&= \begin{pmatrix} h_1^2\mathbb{E}[x_1^2] & h_1h_2\mathbb{E}[x_1x_2] \\ h_1h_2\mathbb{E}[x_1x_2] & h_2^2\mathbb{E}[x_2^2] \end{pmatrix} \\
&= \begin{pmatrix} h_1^2\sigma_1^2 & 0 \\ 0 & h_2^2\sigma_2^2 \end{pmatrix}
\end{aligned}
$$

So each component of the noise has variance proportional to $\sigma_i^2$, and the noise components are independent.

## Minibatch SGD as gradient descent with noise

Now that we understand the noisy gradient, let's see how minibatch SGD works in this setting.

The minibatch SGD update rule is:

$$
w_{k+1} = w_k - \eta \nabla \ell_{B_k}(w_k; x_{B_k})
$$

where $\nabla \ell_{B_k}$ is the average gradient computed over a minibatch $B_k$ of samples.

We can rewrite this as:

$$
\begin{aligned}
w_{k+1} &= w_k - \eta \nabla \ell_{B_k}(w_k; x_{B_k}) \\
&= w_k - \eta (\nabla L(w_k) + \varepsilon_{B_k}) \\
&= (w_k - \eta \nabla L(w_k)) + \eta \varepsilon_{B_k}
\end{aligned}
$$

where $\varepsilon_{B_k}$ is the average noise in the minibatch. Due to the central limit theorem, as the batch size $B$ increases, the variance of this noise decreases proportionally to $1/B$.

Let's look at the component-wise update:

$$
\begin{aligned}
w_{k+1, i} &= w_{k, i} - \eta \nabla \ell_{B_k}(w_{k, i}; x_{B_k})_i \\
&= w_{k, i} - \eta (h_i w_{k, i} - h_i \bar{x}_{B_k, i}) \\
&= (1 - \eta h_i)w_{k, i} + \eta h_i \bar{x}_{B_k, i}
\end{aligned}
$$

where $\bar{x}_{B_k, i}$ is the average of the $i$-th component in the minibatch.

Since $\mathbb{E}[\bar{x}\_{B\_k, i}] = 0$ and $\mathrm{Var}(\bar{x}\_{B\_k, i}) = \sigma_i^2/B$, we can analyze the evolution of the second moment $\mathbb{E}[w_{k, i}^2]$.

Without deriving the full proof, the result is:

$$\boxed{\mathbb{E}[w_{k+1, i}^2] = (1-\eta h_i)^{2k}w_{0, i}^2 + \frac{\eta h_i^2}{B(2-\eta h_i)}\sigma_i^2(1-(1-\eta h_i)^{2k})}$$

This formula has two terms:
1. $(1-\eta h_i)^{2k}w_{0, i}^2$: This term represents how quickly we "forget" the initial value $w_{0, i}$. It decreases exponentially with $k$.
2. $\frac{\eta h_i^2}{B(2-\eta h_i)}\sigma_i^2(1-(1-\eta h_i)^{2k})$: This term represents the steady-state variance due to noise. It approaches $\frac{\eta h_i^2}{B(2-\eta h_i)}\sigma_i^2$ as $k \to \infty$.

We can see that minibatching (increasing $B$) reduces the steady-state variance by a factor of $B$, just as we observed in the one-dimensional case from the previous lecture.

## Higher-dimensional challenges

In the simple one-dimensional case we studied last time, we only had to consider a single convergence rate and a single noise level. In higher dimensions, each component may converge at a different rate and experience different levels of noise.

Let's analyze the trade-offs introduced by having multiple parameters.

### Different convergence rates

Looking at our formula for the expected squared error, we see that the first term, $(1-\eta h_i)^{2k}w_{0, i}^2$, determines how quickly component $i$ converges. Since $h_1 > h_2$ in our setup, we have $(1-\eta h_1) < (1-\eta h_2)$ for a fixed step size $\eta < 1/h_1$. This means that the first component "forgets" its initialization faster than the second component.

The ratio of these two rates depends on the ratio $h_1/h_2$, which is known as the condition number of the problem. The larger this ratio, the more disparate the convergence rates become.

### Different noise levels

The steady-state variance for component $i$ is $\frac{\eta h_i^2}{B(2-\eta h_i)}\sigma_i^2$. This depends on both $h_i$ and $\sigma_i^2$. If the noise level $\sigma_i^2$ varies across dimensions, some components will have higher steady-state variance than others.

### Step size constraints

To ensure convergence, we need $\eta < 2/h_i$ for all $i$. This means the largest eigenvalue (in our case, $h_1$) constrains the maximum stable step size. If $h_1 \gg h_2$, we might be forced to use a very small step size, which will make the second component converge extremely slowly.

### The balancing act

Ideally, we would like to choose a step size $\eta$ that balances the convergence rates and steady-state variances across all dimensions. However, this is generally impossible when the eigenvalues $h_i$ differ significantly.

Specifically, we might try to find $\eta$ such that:

$$
 (1-\eta h_1)^{2k}w_{0, 1}^2 + \frac{\eta h_1^2}{B(2-\eta h_1)}\sigma_1^2(1-(1-\eta h_1)^{2k}) = \\
 (1-\eta h_2)^{2k}w_{0, 2}^2 + \frac{\eta h_2^2}{B(2-\eta h_2)}\sigma_2^2(1-(1-\eta h_2)^{2k})
$$

But this equation generally has no closed-form solution, and may not even have a solution at all, depending on the values of $\sigma_1, \sigma_2, h_1, h_2$, and the initial values $w_0$.

Let's visualize how different parameters affect the convergence of each component:

![Component convergence](figures/component_convergence.png)
*Figure: Convergence of two components with different eigenvalues. Component 1 (with larger $h_1$) converges faster but has higher steady-state variance.*

In the next sections, we'll explore three strategies to mitigate these issues:
1. Momentum: helps with the convergence rate disparity
2. Exponential moving averages: reduces the steady-state variance
3. Preconditioning: addresses both issues by transforming the problem

## Momentum

Momentum is a modification to SGD that incorporates information from past updates. The intuition is that we want to continue moving in the same direction as previous updates, much like a physical object in motion tends to stay in motion.

### Momentum algorithm

The momentum SGD update is:

$$
\begin{aligned}
v_{k+1} &= \beta v_k + \nabla \ell_{B_k}(w_k; x_{B_k}) \\
w_{k+1} &= w_k - \eta v_{k+1}
\end{aligned}
$$

where $\beta \in [0, 1)$ is the momentum parameter.

In our noisy quadratic model, the component-wise update becomes:

$$
\begin{aligned}
v_{k+1, i} &= \beta v_{k, i} + h_i(w_{k, i} - \bar{x}_{B_k, i}) \\
w_{k+1, i} &= w_{k, i} - \eta v_{k+1, i}
\end{aligned}
$$

### How momentum helps

Momentum provides two key benefits:

1. **Accelerated convergence**: For components with small eigenvalues (slow convergence), momentum "accumulates" the gradient over time, effectively increasing the step size in directions of persistent gradient.

2. **Reduced oscillations**: In high-curvature directions, momentum dampens oscillations, allowing for a larger overall step size.

Without deriving the full result, the expected squared error with momentum can be approximated as:

$$\mathbb{E}[w_{k+1, i}^2] \approx \left(1-\frac{\eta h_i}{1-\beta}\right)^{2k}w_{0, i}^2 + \frac{\eta h_i^2}{B(2-\eta h_i)}\frac{1+\beta}{1-\beta}\sigma_i^2$$

for small values of $\eta$ and $k$ large enough.

The first term shows that momentum effectively increases the convergence rate by a factor of $\frac{1}{1-\beta}$. This is especially beneficial for components with small eigenvalues. For example, with $\beta = 0.9$, the effective step size is 10 times larger.

However, the second term shows that momentum also increases the steady-state variance by a factor of $\frac{1+\beta}{1-\beta}$. With $\beta = 0.9$, this means the variance is 19 times higher! This is the price we pay for faster convergence.

### Optimal momentum settings

In practice, the optimal momentum value depends on the condition number of the problem. For problems with large condition numbers (where $h_1 \gg h_2$), higher momentum values (e.g., $\beta = 0.9$ or $0.99$) can significantly accelerate convergence.

However, the increased variance means that momentum is most beneficial when:
1. The batch size $B$ is large (to counteract the variance increase)
2. The condition number is large (so the acceleration benefit outweighs the variance cost)

This explains why momentum often shows little benefit for small batch sizes but significant gains for large batch sizes, as observed in the 2019 paper and in practical deep learning.

Let's visualize the effect of momentum:

![Momentum effect](figures/momentum_effect.png)
*Figure: Effect of momentum on convergence. Momentum accelerates convergence, especially for the slower component (component 2), but increases steady-state variance.*

## Exponential moving averages

Exponential moving average (EMA) is a technique that doesn't modify the optimization algorithm itself but rather the final output. The idea is to maintain a moving average of the parameters during training and use this average for inference.

### EMA algorithm

The EMA update is:

$$
\begin{aligned}
w_{k+1} &= w_k - \eta \nabla \ell_{B_k}(w_k; x_{B_k}) \\
\tilde{w}_{k+1} &= \gamma \tilde{w}_k + (1-\gamma) w_{k+1}
\end{aligned}
$$

where $\gamma \in [0, 1)$ is the averaging coefficient, and $\tilde{w}_k$ is the exponentially averaged parameter vector.

### How EMA helps

EMA provides a simple way to reduce the variance of the final parameters without slowing down convergence. By averaging multiple iterates, EMA "smoothes out" the noise inherent in SGD updates.

Without giving the full derivation, the expected squared error for the EMA parameter is:

$$\mathbb{E}[\tilde{w}_{k+1, i}^2] \approx (1-\eta h_i)^{2k}w_{0, i}^2 + \frac{\eta h_i^2}{B(2-\eta h_i)}\frac{(1-\gamma)(1+(1-\eta h_i)\gamma)}{(1+\gamma)(1-(1-\eta h_i)\gamma)}\sigma_i^2$$

for $k$ large enough and $\gamma < 1-\eta h_i$.

The key observation is that the term $\frac{(1-\gamma)(1+(1-\eta h_i)\gamma)}{(1+\gamma)(1-(1-\eta h_i)\gamma)}$ is strictly less than 1 when $\gamma > 0$. This means EMA reduces the steady-state variance without affecting the convergence rate of the mean.

### Optimal EMA settings

The optimal EMA coefficient $\gamma$ depends on the step size $\eta$ and the eigenvalues $h_i$. A common rule of thumb is to use $\gamma = 0.999$ for large batch sizes and $\gamma = 0.99$ for smaller batch sizes.

One interesting property of EMA is that it's most beneficial when:
1. The batch size is small (high variance in updates)
2. The step size is relatively large (high steady-state variance)

This is exactly the opposite of momentum, which is most beneficial for large batch sizes. This complementary nature makes EMA and momentum a powerful combination in practice.

Let's visualize the effect of EMA:

![EMA effect](figures/ema_effect.png)
*Figure: Effect of exponential moving average. EMA significantly reduces the steady-state variance without slowing down the initial convergence.*

## Preconditioning

Preconditioning addresses the root cause of the convergence rate disparity: the different eigenvalues $h_i$. The idea is to transform the problem so that all dimensions have similar convergence properties.

### Preconditioning algorithm

The preconditioned SGD update is:

$$w_{k+1} = w_k - \eta P^{-1} \nabla \ell_{B_k}(w_k; x_{B_k})$$

where $P$ is a positive definite matrix called the preconditioner.

In our noisy quadratic model, an ideal preconditioner would be $P = \text{diag}(h_1, h_2)$, which would make all eigenvalues equal to 1. However, in practice, we don't know the exact eigenvalues, so we need to approximate $P$.

Let's analyze a family of preconditioners of the form $P = \text{diag}(h_1^p, h_2^p)$ for $0 \leq p \leq 1$. When $p = 0$, we recover standard SGD, and when $p = 1$, we have the ideal preconditioner.

### How preconditioning helps

Preconditioning affects both the convergence rate and the steady-state variance. The component-wise update becomes:

$$w_{k+1, i} = w_{k, i} - \eta h_i^{-p} h_i (w_{k, i} - \bar{x}_{B_k, i}) = w_{k, i} - \eta h_i^{1-p} (w_{k, i} - \bar{x}_{B_k, i})$$

This leads to an expected squared error of:

$$\mathbb{E}[w_{k+1, i}^2] \approx (1-\eta h_i^{1-p})^{2k}w_{0, i}^2 + \frac{\eta h_i^{2-p}}{B(2-\eta h_i^{1-p})}\sigma_i^2$$

The first term shows that preconditioning makes the convergence rates more similar across components. When $p$ is close to 1, the term $(1-\eta h_i^{1-p})$ is approximately the same for all $i$, regardless of $h_i$.

However, the second term shows that preconditioning also affects the steady-state variance. As $p$ increases, the variance term $\frac{\eta h_i^{2-p}}{B(2-\eta h_i^{1-p})}\sigma_i^2$ behaves differently depending on the relationship between $h_i$ and $\sigma_i^2$.

### Optimal preconditioning

The optimal preconditioning strength $p$ depends on the distribution of eigenvalues and noise variances. In practice, popular preconditioners like Adam and K-FAC implicitly estimate the appropriate preconditioning matrix from the observed gradients.

Preconditioning is most beneficial when:
1. The condition number is large (disparity in convergence rates)
2. The batch size is large (so the potential increase in variance is mitigated)

This aligns with empirical findings that preconditioned methods like Adam often outperform vanilla SGD, especially for large batch sizes.

Let's visualize the effect of preconditioning:

![Preconditioning effect](figures/preconditioning_effect.png)
*Figure: Effect of preconditioning. Preconditioning equalizes the convergence rates across components, allowing for faster overall convergence.*

## Experimental comparisons

Now let's compare the different optimization strategies to see how they perform on our noisy quadratic model. We'll create a visualization that shows the convergence behavior for different combinations of techniques.

For our experiments, we'll use the following parameter settings:
- $h_1 = 1.0$, $h_2 = 0.1$ (condition number = 10)
- $\sigma_1^2 = \sigma_2^2 = 1.0$ (equal noise variances)
- Initial values $w_0 = (1.0, 1.0)$
- Various batch sizes: $B \in \{1, 10, 100\}$

We'll compare the following optimization methods:
1. SGD with constant step size
2. SGD with momentum
3. SGD with EMA
4. SGD with preconditioning
5. SGD with momentum and EMA
6. SGD with preconditioning and momentum

Let's analyze the mean squared error (MSE) $\|w_k\|^2$ over iterations for each method.

![Optimization methods comparison](figures/optimization_comparison.png)
*Figure: Comparison of different optimization strategies on the noisy quadratic model. Each line represents a different strategy, and different panels show different batch sizes.*

### Key findings:

1. **SGD**: Basic SGD converges slowly and has high steady-state variance.
2. **SGD+Momentum**: Momentum accelerates convergence but increases variance. It's most beneficial for large batch sizes.
3. **SGD+EMA**: EMA reduces the steady-state variance without affecting the convergence rate. It's particularly helpful for small batch sizes.
4. **SGD+Preconditioning**: Preconditioning equalizes convergence rates, leading to faster overall convergence, especially for large batch sizes.
5. **SGD+Momentum+EMA**: Combining momentum and EMA gives fast convergence with moderate variance, making it effective across all batch sizes.
6. **SGD+Preconditioning+Momentum**: This powerful combination achieves the fastest convergence but requires large batch sizes to control variance.

The results illustrate the trade-offs we've discussed:
- Smaller batch sizes lead to higher variance
- Faster convergence often comes at the cost of higher variance
- Different techniques are complementary and can be combined for better performance

## Conclusion

In this lecture, we've explored stochastic gradient descent through the lens of a noisy quadratic model. We've seen how the basic SGD algorithm can be enhanced with momentum, exponential moving averages, and preconditioning to address the challenges of optimizing multiple parameters.

Key takeaways:

1. **Higher dimensions introduce new challenges**: Different parameters may converge at different rates and experience different levels of noise.

2. **Momentum accelerates convergence**: By accumulating gradients over time, momentum helps overcome the slow convergence of parameters with small eigenvalues, but increases variance.

3. **EMA reduces variance**: Exponential moving averages provide a simple way to reduce the noise in the final parameters without slowing down convergence.

4. **Preconditioning equalizes convergence rates**: By transforming the problem, preconditioning makes all parameters converge at similar rates, allowing for faster overall convergence.

5. **Batch size matters**: The effectiveness of these techniques varies with batch size. Momentum and preconditioning work best with large batches, while EMA is particularly helpful for small batches.

These insights help explain why methods like Adam (which combines momentum and adaptive preconditioning) are so effective in deep learning, especially with large batch sizes. They also explain why techniques like EMA are commonly used to stabilize training.

In our next lecture, we'll see how these concepts extend to more complex optimization problems and explore practical implementations in PyTorch.

## Complete code

Here's a complete script that implements all the optimization methods we've discussed and reproduces the figures from this lecture:

```python
import numpy as np
import matplotlib.pyplot as plt
import torch

# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Define parameters
h1, h2 = 1.0, 0.1  # Eigenvalues
sigma1, sigma2 = 1.0, 1.0  # Noise standard deviations
w0 = np.array([1.0, 1.0])  # Initial parameters
eta = 0.1  # Learning rate
beta = 0.9  # Momentum coefficient
gamma = 0.99  # EMA coefficient
p = 0.5  # Preconditioning power
num_iterations = 1000
batch_sizes = [1, 10, 100]

# Define methods
methods = [
    "SGD",
    "SGD+Momentum",
    "SGD+EMA",
    "SGD+Preconditioning",
    "SGD+Momentum+EMA",
    "SGD+Preconditioning+Momentum"
]

# Colors for plotting
colors = ["blue", "red", "green", "purple", "orange", "brown"]

# Function to generate a batch of samples
def generate_batch(batch_size):
    x1 = np.random.normal(0, sigma1, batch_size)
    x2 = np.random.normal(0, sigma2, batch_size)
    return np.column_stack((x1, x2))

# Function to compute gradient
def compute_gradient(w, batch):
    grad = np.zeros_like(w)
    for i in range(batch.shape[0]):
        grad[0] += h1 * (w[0] - batch[i, 0])
        grad[1] += h2 * (w[1] - batch[i, 1])
    return grad / batch.shape[0]

# Run optimization methods
results = {}

for batch_size in batch_sizes:
    results[batch_size] = {}
    
    for method_idx, method in enumerate(methods):
        # Initialize parameters
        w = w0.copy()
        w_ema = w0.copy()
        v = np.zeros_like(w)
        
        # Track MSE
        mse_history = np.zeros(num_iterations)
        
        for k in range(num_iterations):
            # Generate batch
            batch = generate_batch(batch_size)
            
            # Compute gradient
            grad = compute_gradient(w, batch)
            
            # Apply preconditioning if needed
            if "Preconditioning" in method:
                precond_grad = np.zeros_like(grad)
                precond_grad[0] = grad[0] / (h1 ** p)
                precond_grad[1] = grad[1] / (h2 ** p)
                grad = precond_grad
            
            # Update velocity if using momentum
            if "Momentum" in method:
                v = beta * v + grad
                update = eta * v
            else:
                update = eta * grad
            
            # Update parameters
            w = w - update
            
            # Update EMA if using
            if "EMA" in method:
                w_ema = gamma * w_ema + (1 - gamma) * w
                mse_history[k] = np.sum(w_ema ** 2)
            else:
                mse_history[k] = np.sum(w ** 2)
        
        results[batch_size][method] = mse_history

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Comparison of Optimization Methods on Noisy Quadratic Model", fontsize=16)

for i, batch_size in enumerate(batch_sizes):
    ax = axes[i]
    ax.set_title(f"Batch Size = {batch_size}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean Squared Error")
    ax.set_yscale("log")
    
    for method_idx, method in enumerate(methods):
        ax.plot(results[batch_size][method], color=colors[method_idx], label=method)
    
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.savefig("optimization_comparison.png", dpi=300)
plt.show()

# Generate individual method plots
for method_idx, method in enumerate(methods):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Effect of Batch Size for {method}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean Squared Error")
    ax.set_yscale("log")
    
    for i, batch_size in enumerate(batch_sizes):
        ax.plot(results[batch_size][method], label=f"Batch Size = {batch_size}")
    
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{method.replace('+', '_')}.png", dpi=300)
    plt.close()

# Component-wise convergence analysis
w = w0.copy()
v = np.zeros_like(w)
w_history = np.zeros((num_iterations, 2))

for k in range(num_iterations):
    batch = generate_batch(10)
    grad = compute_gradient(w, batch)
    v = beta * v + grad
    w = w - eta * v
    w_history[k] = w

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Component-wise Convergence with Momentum")
ax.plot(np.abs(w_history[:, 0]), label="Component 1 (h=1.0)")
ax.plot(np.abs(w_history[:, 1]), label="Component 2 (h=0.1)")
ax.set_xlabel("Iterations")
ax.set_ylabel("Absolute Value")
ax.set_yscale("log")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("component_wise_convergence.png", dpi=300)
plt.close()

print("Optimization complete. Figures saved.")
```

This code implements all the optimization methods discussed in the lecture and produces visualizations to compare their performance. You can modify the parameters (h1, h2, sigma1, sigma2, etc.) to explore different problem settings and see how they affect the relative performance of different methods. 