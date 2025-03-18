---
layout: course_page
title: Stochastic Gradient Descent - Insights from a Noisy Quadratic Model
---

# Stochastic Gradient Descent - Insights from a Noisy Quadratic Model

## Table of Contents
1. [Introduction](#introduction)
2. [Problem setup: Noisy Quadratic Model](#problem-setup-noisy-quadratic-model)
3. [Algorithms: EMA, Momentum, and Preconditioning](#algorithms-ema-momentum-and-preconditioning)

## Introduction

In this lecture, we study stochastic optimization through the lens of a simple yet instructive framework: the noisy quadratic model (NQM). Unlike "full-batch" gradient methods, stochastic methods like SGD with *constant step size* do not reach exact minima due to noise in gradient estimates. Instead, these methods converge to a *steady-state risk.* Carefully analyzing this steady-state behavior highlights trade-offs central to stochastic optimization involving the choice of learning rate, batch size, and algorithm variants. We already saw this in the [Lecture 6](../6/notes.md) where we studied the SGD on a simple one-dimensional mean estimation problem. In dimensions greater than one, the behavior is more complex, due to potential "poor conditioning" of the loss function. 

We examine three popular variants of stochastic gradient methods: Exponential Moving Average (EMA), Momentum, and Preconditioning. Each method targets noise reduction and convergence acceleration differently. For example, EMA averages iterates, Momentum averages gradients, and Preconditioning adapts step sizes to problem curvature and see its benefit primarily in higher dimensional problems. 

The results of this lecture are based on the following paper by [Zhang et al. (2019)](https://arxiv.org/pdf/1907.04164). In this paper, the authors provide basic convergence theorems for EMA, momentum, and preconditioning methods. The theoretical results themselves are not new or so surprising. But this is not what is valuable about the paper. Instead, the paper is valuable because it interprets the results of the theorems in terms of the hyperparameters that are common to deep learning training and then *validates* the predictions on such problems. 

In this lecture, we will focus on the statements of the main theoretical results from [Zhang et al. (2019)](https://arxiv.org/pdf/1907.04164). We will not prove the theorems here. We will also not repeat the experiments from the paper. Instead, we will focus on the implications of the theorems for the hyperparameters of SGD. If you're interested in playing around with the experiments from the paper, see [the colab here](https://colab.research.google.com/github/gd-zhang/noisy-quadratic-model/blob/master/nqm.ipynb).

We now introduce the model and the algorithms.

## Problem setup: Noisy quadratic model 

In the [previous lecture](../6/notes.md), we analyzed SGD on a simple one-dimensional mean estimation problem. There we wrote the target loss function $L$ that we wished to optimize as an *expectation* of simpler loss functions: $L(w) = \mathbb{E}\_{i \sim \mathrm{Unif}[1, \ldots, n]}[l(w,x_i)]$. We then estimated the gradient of the loss function ($\nabla L(w)$) by drawing a single sample $x_i$ from the distribution, taking the gradient of the loss corresponding to this sample ($\nabla l(w,x_i)$), and then taking a gradient step in the direction of the sample gradient ($w\_{k+1} = w_k - \alpha \nabla l(w_k, x_i)$). In the process, we proved that this gradient is an *unbiased* estimate of the true gradient of the loss function and that the update $w_{k+1}$ is an *unbiased* estimate of the true gradient step. This unbiasedness was crucial to the analysis that we presented. 

In this lecture, we take a slightly different approach that streamlines the analysis. Instead of writing the target loss function as an expectation and then approximating its gradient with an unbiased random sample, we're going to assume there is a fixed loss function $L$ and that we can compute a "noisy" estimate of it's gradient. We'll work only in dimension 2 and we will assume the loss function is the following simple function 

$$
 L(w) \;=\; \tfrac{1}{2}\big(h_1 w_1^2 + h_2 w_2^2\big)\,, 
$$

where $h_1 \ge h_2 > 0$. Our goal is to optimize this loss function when we can't compute the true gradient of $L$, but only a noisy estimate of it. In that sense its very similar to the mean estimation problem that we studied in the [previous lecture](../6/notes.md). 

What's different here is that we are going to study a variant of sgd that is slightly more general. Namely, we imagine that each SGD step receives a *noisy gradient* $g(w)$ such that 

$$
 g(w) \;=\; \nabla L(w) + \xi \;=\; (h_1 w_1,\, h_2 w_2) + \xi\,, 
$$
 
where $\mathbb{E}[\xi]=0$ and $\operatorname{Cov}(\xi)=\mathrm{diag}(c\_1/B,\,c\_2/B)$. In other words, each component of the gradient is corrupted by uncorrelated noise of variance $c\_i/B$. Where are $c\_i$ and $B$ coming from? Let me try to explain briefly in the following blurb: 
> The "noisy gradient" $g$ is an unbiased estimate of the true gradient $\mathbb{E}[g(w)] = \nabla L(w)$. How does it arise? Let's say that we have $B$ unbiased samples $G\_1, \ldots, G\_B$ of $\nabla L(w)$, meaning $\mathbb{E}[G\_i] = \nabla L(w)$ for each i. Let's suppose further that each $G_i$ has covariance matrix $\mathrm{diag}(c\_1,\,c\_2)$. Then we can think of $g$ as an average of these noisy gradients: $g(w) = \frac{1}{B}\sum\_{i=1}^B G\_i(w)$. One can then quickly check that $\xi = \nabla L(w) - g(w)$ satisfies $\mathbb{E}[\xi]=0$ and $\operatorname{Cov}(\xi)=\mathrm{diag}(c\_1/B,\,c\_2/B)$. Thus, we get $g$ from averaging a *batch* of size $B$ of sample gradients and the coordinates of each sample gradient have variance $c_i$.

Because of the above, we think of the quantity $B$ in the definition of $g$ and $\xi$ as a batch size and throughout we consider the effect of $B$ on algorithm performance. 

Before moving to algorithms, let me point out that we are going to be a bit sloppy with notation: For each $k$ and $B$ there will be a different noisy gradient $g_{k, B}(w)$ because the noise is different at each step and depends on how we set $B$. However, for simplicity, we will drop the subscripts and just write $g(w)$ from here on. We will think of $B$ as a knob we can turn throughout our analysis, but we cannot change $c_i$.

### SGD on the NQM

To apply SGD in this setting, we fix a stepsize $\alpha$ (often called a "learning rate") and update the iterates at time-step $k$ as follows: 

$$
 w_{k+1,i} = w_{k,i} - \alpha\, g_i(w_k) = w_{k,i} - \alpha\big(h_i w_{k,i} + \xi_{k,i}\big)\,. 
$$
 
Substituting the noise term and taking expectations, one finds a simple *linear recurrence* with batch size $B$. Treating $w_{k,i}$ as a random variable, its mean and variance evolve as:

$$
 
\mathbb{E}[w_{k+1,i}] = (1 - \alpha h_i)\,\mathbb{E}[w_{k,i}]\,, 
\qquad 
\mathbb{V}[w_{k+1,i}] = (1 - \alpha h_i)^2\,\mathbb{V}[w_{k,i}] + \frac{\alpha^2\,c_i}{B}\,. 

$$

Here the quantity $\mathbb{V}$ denotes the variance of a random variable (we adopt this notation to stay consistent with the NQM paper). 

The statement and proof of the above result is nearly identical to those of [the previous lecture](../6/notes.md). In particular, as before, these equations say that in expectation, $w_i$ decays exponentially by a factor $(1-\alpha h_i)$ each step toward the optimum. In contrast, the variance of $w_i$ does not decay exponentially, because at each step noise is injected at a rate proportional to $\alpha^2/B$. The consequence of this is that this process has a **steady-state**: as $k\to\infty$, $\mathbb{E}[w_{k,i}] \to 0$ but $\mathbb{V}[w_{k,i}]$ converges to a constant. 


In the last lecture, we mainly concerned ourselves with how quickly the iterates $w_k$ converged to the optimum. Here, we are interested in how quickly the **loss** $L(w_k)$ converges to the minimum value $L(0)=0$. We can derive the expected **loss** (or "risk") in dimension $i$ (for $i \in \{1, 2\}$) at step $k$ as 

$$
 
\mathbb{E}[\ell_i(w_{k,i})] = (1-\alpha h_i)^{2k} \mathbb{E}[\ell_i(w_{0,i})] + (1-(1-\alpha h_i)^{2k}) \frac{\alpha c_i}{2B(2-\alpha h_i)},

$$

where $\ell_i(w_i) = \frac{1}{2}h_i w_i^2$ and $L(w) = \ell_1(w_1) + \ell_2(w_2)$.  
 
After many steps ($k$ large enough that $(1-\alpha h_i)^{2k}\approx0$), the first term nearly vanishes and the loss approaches a **steady-state risk**: 

$$
 L^{\text{(ss)}}_i \;=\; \frac{\alpha c_i}{2B(2-\alpha h_i)}\,. 
$$
 
This residual loss is the price we pay for using a positive learning rate with noisy gradients—SGD will hover around the optimum but can't converge perfectly due to the noise. Notice that $L^{\text{(ss)}}_i$ is **smaller** when the batch size $B$ is larger (averaging out more noise) and *larger* when the learning rate $\alpha$ is larger. In fact, for each coordinate $i$ there is a trade-off: a higher $\alpha$ gives faster initial decay (since $(1-\alpha h_i)^{2k}$ vanishes quicker) but yields a higher noise floor $L^{\text{(ss)}}_i$. 

### The difficulty with higher dimensions: large conditioning

In previous lectures we introduced the "condition number" of a linear system as the ratio of the largest to smallest singular values of a matrix. We saw that when a linear system has a large condition number, direct methods and iterative methods like gradient descent struggle: see [lecture 2](../2/notes.md) and [lecture 3](../3/notes.md). Beyond least squares, it turns out optimization problems can also have condition numbers, too. For example, in the context of NQM, the condition number of the problem is defined as the quantity:

$$
\kappa = \frac{h_1}{h_2}
$$

This $\kappa$ is called the *condition number* of the problem because it is the condition number of the Hessian $\nabla^2 L(w) = \mathrm{diag}(h_1, h_2)$ of the loss function. In general, whenever the Hessian of the loss function has a large condition number, SGD will slow down. Why might this be? 

To see why, notice that in order for the the first term in the loss recurrence to eventually vanish

$$
\mathbb{E}[\ell_i(w_{k,i})] = (1-\alpha h_i)^{2k} \mathbb{E}[\ell_i(w_{0,i})] + (1-(1-\alpha h_i)^{2k}) \frac{\alpha c_i}{2B(2-\alpha h_i)},
$$

we must have that $(1-\alpha h_i)^{2k}$ is smaller than 1 for each $i$. In particular, we need 

$$
\alpha \leq \min\{2/h_1, 2/h_2\} = 2/h_1,
$$

where the final inequality follows because $h_1 \geq h_2$. Thus, when we run SGD with a large, but allowable constant stepsize $\alpha = 1/h_1$, the convergence rate of the second component $\mathbb{E}[\ell_2(w_{k,2})]$ is 

$$
 (1-\alpha h_2)^{2k} = (1 - 1/\kappa)^{2k}.
$$

In particular, if $\kappa$ is large, then the convergence rate of the second component can be extremely slow. 

## Algorithms: EMA, Momentum, and Preconditioning

In the following sections, we introduce three techniques that help improve the performance of SGD: exponential moving average (EMA), momentum, and preconditioning. If you're busy and you want to get the TL;DR: 

- EMA helps with small batch sizes by reducing the steady-state risk without sacrificing the initialization error.
- Momentum and preconditioning both help with large batch sizes by reducing the initialization error. However, they also amplify the steady-state risk, so we need to be careful when batch sizes are low.

We now turn to the EMA.

### EMA 

EMA is a modification of SGD that reduces its steady state risk *without* sacrificing convergence rate. It is help mostly when the batch-size of SGD is small. The EMA algorithm does not actually modify the update rule of SGD. Instead, it only changes where we evaluate the loss function. Specifically, we now maintain two iterates, defined precisely as:

$$

w_{k+1} = w_{k} - \alpha\, g(w_k), \quad \tilde{w}_{k+1} = \gamma \tilde{w}_{k} + (1 - \gamma) w_{k+1}

$$

The sequence $w_k$ is precisely the SGD sequence. On the other hand, the sequence $\tilde{w}_k$ is a simple *exponential moving average* of the SGD sequence. This scheme has minimal computational and memory overhead, requiring only one additional copy of parameters and simple arithmetic averaging.

EMA's advantage can be gleaned from Theorem 2 of [Zhang et al. (2019)](https://arxiv.org/pdf/1907.04164), which states that 

$$
\begin{aligned}
\mathbb{E}[\ell(\tilde{w}_{k,i})] &\leq \left(\frac{(r_1^{k+1}-r_2^{k+1}) - \gamma(1-\alpha h_i)(r_1^k - r_2^k)}{r_1 - r_2}\right)^2 \mathbb{E}[\ell(w_{0,i})]\\
&\hspace{20pt}+ \frac{\alpha c_i}{2B(2-\alpha h_i)}\frac{(1-\gamma)(1 + (1-\alpha h_i)\gamma)}{(1+\gamma)(1-(1-\alpha h_i)\gamma)},
\end{aligned}
$$

where $ r_1 = 1-\alpha h_i $ and $ r_2 = \gamma $.

The theorem shows that when $\gamma < 1 - \alpha \min_i\{h_i\}$, EMA reduces steady-state risk without sacrificing convergence rate. Explicitly, because $r_1 = 1 - \alpha h_i$ and $r_2 = \gamma < r_1$, it follows that $r_1 > r_2$. Thus, for large $k$, terms involving $r_2^k$ become negligible relative to those involving $r_1^k$, and the initialization error contraction factor approximates:

$$

\frac{(r_1^{k+1}-r_2^{k+1}) - \gamma(1-\alpha h_i)(r_1^k - r_2^k)}{r_1 - r_2} \approx r_1^k,

$$

showing the convergence rate matches that of plain SGD, namely $(1 - \alpha h_i)^{2k}$.

Thus, in the *small batch-size regime*, where steady-state risk is large for SGD, EMA can outperform SGD without sacrificing convergence rate. However, in the *large batch-size regime*, where initialization error dominates steady-state risk, EMA provides minimal additional benefit over standard SGD. This explicitly clarifies EMA's primary role: reducing steady-state risk, rather than initialization error.

### Momentum

We now analyze Momentum within the NQM framework. Unlike EMA, which averages iterates, Momentum averages past gradients to reduce noise and accelerate convergence. Precisely matching our existing notation, the Momentum algorithm updates as:

$$

m_{k+1,i} = \beta m_{k,i} + g_i(w_k), \quad w_{k+1,i} = w_{k,i} - \alpha m_{k+1,i}

$$


Note the parallels with EMA: both involve exponential averaging, but EMA averages parameter iterates, whereas Momentum averages gradients. Critically, unlike EMA, Momentum modifies the parameter update rule directly.

The advantage of Momentum is clearly established by Theorem 1 from Zhang et al. (2019):

$$

\mathbb{E}[\ell(w_{k,i})] \leq \left(\frac{(r_1^{k+1}-r_2^{k+1}) - \beta(r_1^k - r_2^k)}{r_1 - r_2}\right)^2 \mathbb{E}[\ell(w_{0,i})] + \frac{(1 + \beta)\alpha c_i}{2B(2\beta + 2 - \alpha h_i)(1 - \beta)},

$$

where $ r\_1 $ and $ r\_2 $ are the two roots of the quadratic equation:

$$

z^2 - (1 - \alpha h_i + \beta)z + \beta = 0.

$$


The convergence rate explicitly depends on $ r_1 $, and we must distinguish clearly between two regimes based on the roots:

1. **Overdamped regime** ($\beta < (1-\sqrt{\alpha h_i})^2$, leading toreal roots $r_1$, $r_2$): here, the convergence rate is dominated by the larger root, approximately 
$r_1 \approx 1 - \frac{\alpha h_i}{1-\beta}$. 
Thus, increasing $\beta$ reduces $|r_1|$, accelerating convergence.

2. **Underdamped regime** ($\beta \geq (1-\sqrt{\alpha h_i})^2$, leading to complex roots $r_1$, $r_2$): characterized by complex conjugate roots where $|r_1| = |r_2| = \sqrt{\beta}$. 
In this regime, increasing $\beta$ does not further improve the convergence rate, as the magnitude is fixed near $\sqrt{\beta}$. 
However, increasing $\beta$ significantly amplifies the steady-state risk by a factor of approximately $\frac{1}{1-\beta}$. 
Thus, the optimal choice of $\beta$ lies at or just below the boundary between these two regimes.

Momentum thus provides two key impacts:
- **Reduces initialization error** by accelerating the initial convergence.
- **Increases steady-state risk** by amplifying it by approximately a factor of $\frac{1}{1-\beta}$.

These trade-offs make Momentum beneficial in the large batch-size regime, where steady-state risk is inherently small, and initialization error dominates. Conversely, for small batch sizes—where steady-state risk dominates—the benefits of Momentum are minimal compared to standard SGD. Momentum's advantage is most pronounced with large batch sizes.


### Preconditioning

We now analyze Preconditioning within the NQM framework. Precisely matching the established notation, the preconditioning algorithm is defined as:

$$

w_{k+1,i} = w_{k,i} - \alpha h_i^{-p} g_i(w_k)

$$


Preconditioning aims to **balance convergence rates** across different dimensions by employing dimension-specific learning rates. Dimensions with smaller curvature $h_i$ typically converge slowest under standard SGD. By using preconditioning with factor $h_i^{-p}$, convergence is accelerated in these slower dimensions. However, this beneficial balancing of convergence rates also significantly **increases steady-state risk**. 

Indeed, results from Section 3.3 of Zhang et al. (2019) provide show that:

- **Component-wise Loss Bound:**  
  
$$

  \mathbb{E}[\ell_i(w_{k,i})] \leq (1 - \alpha h_i^{1-p})^{2k} \mathbb{E}[\ell_i(w_{0,i})] + \frac{\alpha c_i h_i^{-p}}{2B(2 - \alpha h_i^{1-p})}
  
$$


- **Convergence Rate:**  
  The convergence rate changes from $1 - \alpha h_i$ (without preconditioning) to $1 - \alpha h_i^{1-p}$. As $p$ grows, dimensions with smaller curvature $h_i$ converge faster, since $h_i^{1-p}$ increases when $h_i$ is small.

- **Steady-state Risk:**  
  The steady-state risk increases from $\frac{\alpha c_i}{2B(2 - \alpha h_i)}$ (without preconditioning) to approximately:
  
$$

  \frac{\alpha c_i h_i^{-p}}{2B(2 - \alpha h_i^{1-p})}
  
$$

  Increasing $p$ magnifies steady-state risk, particularly in dimensions with low curvature.

Hence, with large batch sizes, the optimal choice for $p$ is close to 1, balancing faster convergence in slow dimensions against increased steady-state risk. Therefore, preconditioning is most beneficial in the **large batch-size regime**, where steady-state risk is naturally lower, and the primary objective is rapid reduction of initialization error. Conversely, in the small batch-size regime—characterized by higher steady-state risk—preconditioning can negatively affect performance compared to plain SGD.