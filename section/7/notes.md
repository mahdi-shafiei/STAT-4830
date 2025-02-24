---
layout: course_page
title: Stochastic Gradient Descent - Insights from a Noisy Quadratic Model
---

# Stochastic Gradient Descent: Insights from a Noisy Quadratic Model

## Table of contents
1. [Introduction](#introduction)
2. [Problem setup: Noisy quadratic model](#problem-setup-noisy-quadratic-model)
3. [SGD through the lens of the noisy quadratic model](#sgd-through-the-lens-of-the-noisy-quadratic-model)
4. [Higher-dimensional challenges](#higher-dimensional-challenges)
5. [Momentum](#momentum)
6. [Exponential moving averages](#exponential-moving-averages)
7. [Preconditioning](#preconditioning)
8. [Experimental comparisons](#experimental-comparisons)
9. [Conclusion](#conclusion)

## Introduction

In our previous lecture, we examined stochastic gradient descent (SGD) in the context of large-scale optimization problems. We saw that SGD offers a memory-efficient approach for problems with massive datasets by sampling random data points rather than processing the entire dataset at once. We analyzed its convergence properties, both in expectation and variance, and observed the trade-offs involved in choosing the step size.

Today, we'll extend this understanding by studying SGD through the lens of a noisy quadratic model (NQM). While simple, this model captures many of the essential behaviors we see in real neural network optimization, making it a valuable tool for generating testable predictions. The insights we gain will help us understand several practical SGD modifications used in deep learning:

1. **Momentum**: How carrying velocity from previous updates can improve convergence
2. **Exponential moving averages**: How averaging parameter values can reduce variance
3. **Preconditioning**: How reshaping the optimization landscape can accelerate convergence

These modifications are crucial for training deep neural networks efficiently, helping to navigate complex, high-dimensional loss landscapes. Our analysis is inspired by the 2019 paper "Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model" by researchers at Google, DeepMind, University of Toronto, and Anthropic.

## Problem setup: Noisy quadratic model

Let's begin with a simple multivariate quadratic problem. Fix constants $h_1 > h_2 > 0$ (eigenvalues of our quadratic loss surface) and $\sigma_1, \sigma_2 > 0$ (noise standard deviations). Consider a multivariate random variable $x \in \mathbb{R}^2$, satisfying:

- Mean zero: $\mathbb{E}[x_1] = \mathbb{E}[x_2] = 0$
- Independent components: $\mathbb{E}[x_1x_2] = \mathbb{E}[x_1]\mathbb{E}[x_2] = 0$
- Known variances: Each component has variance $\sigma_i^2$, meaning $\mathbb{E}[x_i^2] = \sigma_i^2$ for $i = 1, 2$

Our loss function is a simple quadratic:

$$
L(w) = \frac{1}{2}\mathbb{E}_{(x_1, x_2) \sim P}\left[h_1(x_1 - w_1)^2 + h_2(x_2 - w_2)^2\right]
$$

This resembles the model we worked with in the previous lecture, but now we have two parameters $w_1$ and $w_2$ instead of just one. We can define the sample-wise loss for a specific datapoint $(x_1, x_2)$:

$$
\ell(w; (x_1, x_2)) = \frac{1}{2}\left[h_1(x_1 - w_1)^2 + h_2(x_2 - w_2)^2\right]
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

Let's compute the expectation of the gradient of the sample loss:

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

We can express the stochastic gradient as the true gradient plus noise:

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

## SGD through the lens of the noisy quadratic model

Now that we understand the noisy gradient, let's analyze how minibatch SGD works in this setting.

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
&= (w_k - \eta \nabla L(w_k)) - \eta \varepsilon_{B_k}
\end{aligned}
$$

where $\varepsilon_{B_k}$ is the average noise in the minibatch. Due to the central limit theorem, as the batch size $B$ increases, the variance of this noise decreases proportionally to $1/B$.

Looking at the component-wise update:

$$
\begin{aligned}
w_{k+1, i} &= w_{k, i} - \eta \nabla \ell_{B_k}(w_{k, i}; x_{B_k})_i \\
&= w_{k, i} - \eta (h_i w_{k, i} - h_i \bar{x}_{B_k, i}) \\
&= (1 - \eta h_i)w_{k, i} + \eta h_i \bar{x}_{B_k, i}
\end{aligned}
$$

where $\bar{x}_{B_k, i}$ is the average of the $i$-th component in the minibatch.

Since $\mathbb{E}[\bar{x}_{B_k, i}] = 0$ and $\text{Var}(\bar{x}_{B_k, i}) = \sigma_i^2/B$, we can analyze the evolution of the second moment $\mathbb{E}[w_{k, i}^2]$.

The expected risk after $t$ steps in a given dimension $i$ is:

$$\boxed{\mathbb{E}[\ell(\theta_i(t))] = \underbrace{(1 - \eta h_i)^{2t}}_{\text{convergence rate}} \mathbb{E}[\ell(\theta_i(0))] + \underbrace{\frac{\eta h_i^2\sigma_i^2}{2B(2-\eta h_i)}}_{\text{steady state risk}} (1-(1-\eta h_i)^{2t})}$$

where we've assumed $\eta h_i \leq 2$ for stability.

This formula has two key terms:
1. **Convergence term**: $(1-\eta h_i)^{2t}\mathbb{E}[\ell(\theta_i(0))]$ - This represents how quickly we "forget" the initial value. It decreases exponentially with $t$.
2. **Steady-state risk term**: $\frac{\eta h_i^2\sigma_i^2}{2B(2-\eta h_i)}(1-(1-\eta h_i)^{2t})$ - This represents the asymptotic variance due to noise. It approaches $\frac{\eta h_i^2\sigma_i^2}{2B(2-\eta h_i)}$ as $t \to \infty$.

We can see that minibatching (increasing $B$) reduces the steady-state variance by a factor of $B$, just as we observed in the one-dimensional case from the previous lecture.

## Higher-dimensional challenges

In the one-dimensional case we studied previously, we only had to consider a single convergence rate and a single noise level. In higher dimensions, each component may converge at a different rate and experience different levels of noise. This introduces several challenges:

### Different convergence rates

Looking at our formula for the expected squared error, we see that the first term, $(1-\eta h_i)^{2t}\mathbb{E}[\ell(\theta_i(0))]$, determines how quickly component $i$ converges. Since $h_1 > h_2$ in our setup, we have $(1-\eta h_1) < (1-\eta h_2)$ for a fixed step size $\eta < 1/h_1$. This means that the first component (corresponding to the larger eigenvalue) "forgets" its initialization faster than the second component.

The condition number of the problem, defined as the ratio $\kappa = h_1/h_2$, determines how disparate these convergence rates are. The larger the condition number, the more the convergence rates differ across dimensions.

### Different noise levels

The steady-state variance for component $i$ is $\frac{\eta h_i^2\sigma_i^2}{2B(2-\eta h_i)}$. This depends on both $h_i$ and $\sigma_i^2$. If the noise level $\sigma_i^2$ varies across dimensions, some components will have higher steady-state variance than others.

### Step size constraints

To ensure convergence, we need $\eta < 2/h_i$ for all $i$. This means the largest eigenvalue (in our case, $h_1$) constrains the maximum stable step size. If $h_1 \gg h_2$, we might be forced to use a very small step size, which will make the second component converge extremely slowly.

### The balancing act

Ideally, we would choose a step size $\eta$ that balances the convergence rates and steady-state variances across all dimensions. However, this is generally impossible when the eigenvalues $h_i$ differ significantly.

In the next sections, we'll explore three strategies to mitigate these issues:
1. Momentum: helps with the convergence rate disparity
2. Exponential moving averages: reduces the steady-state variance
3. Preconditioning: addresses both issues by transforming the problem

## Momentum

Momentum is a modification to SGD that incorporates information from past updates. The intuition is that we want to continue moving in directions of persistent gradients, much like a physical object in motion tends to stay in motion.

### Momentum algorithm

The momentum SGD update is:

$$
\begin{aligned}
v_{k+1} &= \beta v_k + \nabla \ell_{B_k}(w_k; x_{B_k}) \\
w_{k+1} &= w_k - \eta v_{k+1}
\end{aligned}
$$

where $\beta \in [0, 1)$ is the momentum parameter and $v_k$ is the velocity. In our noisy quadratic model, the component-wise update becomes:

$$
\begin{aligned}
v_{k+1, i} &= \beta v_{k, i} + h_i(w_{k, i} - \bar{x}_{B_k, i}) \\
w_{k+1, i} &= w_{k, i} - \eta v_{k+1, i}
\end{aligned}
$$

### Momentum dynamics theorem

According to the paper, the following result holds for momentum SGD:

**Theorem 1:** Given a dimension index $i$, and $0 \leq \beta < 1$ with $\beta \neq (1 - \sqrt{\alpha h_i})^2$, the expected risk at time $t$ associated with that dimension satisfies the upper bound:

$$
\mathbb{E}[\ell(\theta_i(t))] \leq \left(\frac{(r_1^{t+1} - r_2^{t+1}) - \beta(r_1^t - r_2^t)}{r_1 - r_2}\right)^2 \mathbb{E}[\ell(\theta_i(0))] + \frac{(1+\beta)\eta h_i^2\sigma_i^2}{2B(2\beta + 2 - \eta h_i)(1-\beta)}
$$

where $r_1$ and $r_2$ (with $r_1 \geq r_2$) are the two roots of the quadratic equation $x^2 - (1-\eta h_i + \beta)x + \beta = 0$.

As with plain SGD, the loss for each dimension can be expressed as the sum of two terms:
1. A term that decays exponentially, corresponding to the behavior of the deterministic version of the algorithm.
2. A constant term representing the steady-state risk due to noise.

### How momentum helps

Momentum provides two key benefits:

1. **Accelerated convergence**: For small learning rates, the convergence rate with momentum is approximately $(1-\frac{\eta h_i}{1-\beta})$, compared to $(1-\eta h_i)$ without momentum. This means momentum effectively increases the learning rate by a factor of $\frac{1}{1-\beta}$. For components with small eigenvalues (slow convergence), this acceleration is particularly beneficial.

2. **Damped oscillations**: In high-curvature directions, momentum dampens oscillations, allowing for a larger overall step size.

However, momentum also increases the steady-state variance by a factor of approximately $\frac{1+\beta}{1-\beta}$. With $\beta = 0.9$ (a common value), this means the variance is about 19 times higher! This is the price we pay for faster convergence.

![Momentum convergence](figures/momentum.pdf)
*Figure: Convergence rate and steady state risk as a function of momentum for a single dimension with $\alpha h = 0.0005$. Higher momentum values accelerate convergence but increase steady-state variance.*

### When momentum helps most

Momentum is most beneficial when:
1. The batch size $B$ is large (to counteract the variance increase)
2. The condition number is large (so the acceleration benefit outweighs the variance cost)

This explains why momentum often shows little benefit for small batch sizes but significant gains for large batch sizes, as observed in practice. As shown in Figure 1 from the paper, momentum SGD (solid lines) has no benefit over plain SGD (dashed lines) at small batch sizes, but extends the perfect scaling to larger batch sizes.

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

### EMA dynamics theorem

According to the paper, the following result holds for EMA:

**Theorem 2:** Given a dimension index $i$, and $0 \leq \gamma < 1$, the expected risk at time $t$ associated with that dimension satisfies the upper bound:

$$
\begin{aligned}
\mathbb{E}[\ell(\tilde{\theta}_i(t))] \leq &\left(\frac{(r_1^{t+1} - r_2^{t+1}) - \gamma(1-\alpha h_i)(r_1^t - r_2^t)}{r_1 - r_2}\right)^2 \mathbb{E}[\ell(\theta_i(0))] \\
&+ \frac{\alpha h_i^2\sigma_i^2}{2B(2-\alpha h_i)} \frac{(1-\gamma)(1+(1-\alpha h_i)\gamma)}{(1+\gamma)(1-(1-\alpha h_i)\gamma)}
\end{aligned}
$$

where $r_1 = 1-\eta h_i$ and $r_2 = \gamma$.

### How EMA helps

EMA reduces the steady-state variance without affecting the convergence rate of the mean. By properly choosing an averaging coefficient $\gamma < 1 - \alpha h_d$ (so that $r_1 > r_2$), the colored term in Theorem 2 becomes strictly less than 1. This means EMA reduces the steady-state risk compared to plain SGD, without sacrificing convergence speed.

![EMA effect](figures/ema-nqm.pdf)
*Figure: Effects of exponential moving average. Solid lines are SGD with EMA while dashed lines are plain SGD. EMA reduces the number of steps required, especially at small batch sizes.*

### When EMA helps most

EMA is most beneficial when:
1. The batch size is small (high variance in updates)
2. The step size is relatively large (high steady-state variance)

This is complementary to momentum, which works best for large batch sizes. The figure shows that EMA reduces the steps required, especially for plain SGD, and becomes redundant in the large-batch regime. Another important observation is that EMA reduces the critical batch size, allowing the same acceleration with less computation.

## Preconditioning

Preconditioning addresses the root cause of the convergence rate disparity: the different eigenvalues $h_i$. The idea is to transform the problem so that all dimensions have similar convergence properties.

### Preconditioning algorithm

The preconditioned SGD update is:

$$w_{k+1} = w_k - \eta P^{-1} \nabla \ell_{B_k}(w_k; x_{B_k})$$

where $P$ is a positive definite matrix called the preconditioner.

In our noisy quadratic model, an ideal preconditioner would be $P = \text{diag}(h_1, h_2)$, which would make all eigenvalues equal to 1. However, in practice, we don't know the exact eigenvalues, so we need to approximate $P$.

The paper analyzes a family of preconditioners of the form $P = \text{diag}(h_1^p, h_2^p)$ for $0 \leq p \leq 1$. When $p = 0$, we recover standard SGD, and when $p = 1$, we have the ideal preconditioner.

### Preconditioning dynamics

The component-wise update with preconditioning is:

$$w_{k+1, i} = w_{k, i} - \eta h_i^{-p} h_i (w_{k, i} - \bar{x}_{B_k, i}) = w_{k, i} - \eta h_i^{1-p} (w_{k, i} - \bar{x}_{B_k, i})$$

For the NQM, the dynamics of preconditioned SGD are equivalent to the SGD dynamics in a transformed problem with Hessian $\tilde{H} = P^{-1/2}HP^{-1/2}$ and gradient covariance $\tilde{C} = P^{-1/2}CP^{-1/2}$. 

The expected risk with preconditioning becomes:

$$\mathbb{E}[L(w(t))] \leq \sum_{i=1}^d (1-\eta h_i^{1-p})^{2t} \mathbb{E}[\ell(\theta_i(0))] + \sum_{i=1}^d \frac{\eta h_i^{2-p}\sigma_i^2}{2B(2-\eta h_i^{1-p})}$$

### How preconditioning helps

Preconditioning has two key effects:

1. **Equalized convergence rates**: As $p$ approaches 1, the terms $(1-\eta h_i^{1-p})$ become more similar across dimensions, regardless of the original eigenvalues $h_i$. This means all components converge at roughly the same rate.

2. **Transformation of the steady-state risk**: Preconditioning changes the steady-state risk term. For ill-conditioned problems (where $h_1 \gg h_2$), the steady-state risk becomes approximately $\frac{h_i^{2-p}\sigma_i^2}{2Bh_1}\frac{(h_i/h_1)^{-p}}{1-(h_i/h_1)^{1-p}}$, which increases with $p$.

![Preconditioning powers](figures/sgd_momentum.pdf)
*Figure: Effects of momentum and preconditioning. Steps required to reach target loss as a function of batch size under different preconditioning power. Solid lines are momentum SGD while dashed lines are plain SGD. The black dashed line is the information theoretic lower bound.*

### When preconditioning helps most

Preconditioning is most beneficial when:
1. The condition number is large (disparity in convergence rates)
2. The batch size is large (so the potential increase in variance is mitigated)

This aligns with empirical findings that preconditioned methods like Adam and K-FAC often outperform vanilla SGD, especially for large batch sizes. The figure shows that higher preconditioning powers extend perfect scaling to larger batch sizes, and preconditioning combined with momentum is particularly effective.

## Experimental comparisons

The paper validates these theoretical insights on real neural networks. Let's examine some key findings from their experiments:

![Large-scale experiments](figures/large-scale.png)
*Figure: Empirical relationship between batch size and steps to result for various neural network architectures and optimization methods. Key observations: 1) momentum SGD has no benefit over plain SGD at small batch sizes, but extends the perfect scaling to larger batch sizes; 2) preconditioning also extends perfect scaling to larger batch sizes, i.e., K-FAC > Adam > momentum SGD; 3) preconditioning (particularly K-FAC) reduces the number of steps needed to reach the target even for small batch sizes.*

Each optimizer shows two distinct regimes: a small-batch (stochastic) regime with perfect linear scaling, and a large-batch (deterministic) regime insensitive to batch size. The transition between these regimes is called the critical batch size.

The results confirm the predictions from the noisy quadratic model:

1. **Effect of momentum**: Momentum-based optimizers match plain SGD methods in the small-batch regime but give substantial speedups in the large-batch regime.

2. **Effect of preconditioning**: Preconditioning increases the critical batch size and gives substantial speedups in the large-batch regime, but also improves performance by a small constant factor even for very small batches.

3. **Effect of EMA**: EMA reduces the number of steps required, especially for plain SGD, and becomes redundant in the large-batch regime. EMA reduces the critical batch size, allowing the same acceleration with less computation.

The paper also explores optimal learning rates and learning rate schedules, finding that the optimal learning rate scales linearly with batch size before reaching the critical batch size, consistent with common practices in deep learning.

## Conclusion

In this lecture, we've explored stochastic gradient descent through the lens of a noisy quadratic model. We've seen how the basic SGD algorithm can be enhanced with momentum, exponential moving averages, and preconditioning to address the challenges of optimizing multiple parameters.

Key takeaways:

1. **Higher dimensions introduce new challenges**: Different parameters may converge at different rates and experience different levels of noise.

2. **Momentum accelerates convergence**: By accumulating gradients over time, momentum helps overcome the slow convergence of parameters with small eigenvalues, but increases variance. It's most beneficial for large batch sizes.

3. **EMA reduces variance**: Exponential moving averages provide a simple way to reduce the noise in the final parameters without slowing down convergence. It's particularly helpful for small batch sizes.

4. **Preconditioning equalizes convergence rates**: By transforming the problem, preconditioning makes all parameters converge at similar rates, allowing for faster overall convergence.

5. **Batch size matters**: The effectiveness of these techniques varies with batch size. Momentum and preconditioning work best with large batches, while EMA is particularly helpful for small batches.

These insights help explain why methods like Adam (which combines momentum and adaptive preconditioning) and K-FAC are so effective in deep learning, especially with large batch sizes. They also explain why techniques like EMA are commonly used to stabilize training.

The noisy quadratic model, despite its simplicity, captures many of the essential phenomena in neural network optimization. It provides a framework for understanding and predicting the behavior of different optimization algorithms across various batch sizes, making it a valuable tool for both researchers and practitioners. 