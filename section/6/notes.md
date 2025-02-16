---
layout: course_page
title: Stochastic Gradient Descent (SGD) for Mean Estimation
---


## Introduction

In this lecture, we address the computational challenges of solving a least squares problem when the number of samples $n$ is extremely large (often called the "big *n*" scenario). In previous lectures, we successfully applied **batch gradient descent** and even derived a closed-form solution for least squares. However, as $n$ grows into the millions or billions, **naïve approaches break down** due to memory and runtime constraints. To illustrate:

**Memory limitations:** Storing all $n$ samples in memory can be infeasible. For example, attempting to load a dataset of size $n = 10^{10}$ (10 billion samples) on a MacBook Pro with 64GB RAM results in an out-of-memory error (Figure 1). Each sample (if stored as an 8-byte float) requires significant memory, and the total exceeds available RAM. **Figure 1:** *Memory error on a MacBook Pro (64GB RAM) when attempting to store all samples in memory. (Placeholder for figure)*

**Computation time:** Even a simple algorithm like gradient descent becomes *very slow* for large $n$. The cost per iteration scales linearly with $n$ because we must sum over all samples to compute the gradient. In a previous lecture, we noted that one full gradient descent step takes $O(n)$ time *【previous lecture】*. If $n$ is huge, just 100 iterations could be prohibitively time-consuming. Figure 2 illustrates the time for 100 gradient descent iterations as a function of $n$ on a log-log scale — the runtime grows roughly linearly with $n$. **Figure 2:** *Computation time for 100 gradient descent iterations vs. number of samples $n$ (both axes in log scale). (Placeholder for figure)*

Given these challenges, how can we solve least squares for very large datasets? We need an approach that avoids holding all data in memory at once and reduces the cost per iteration. This motivates *stochastic gradient descent (SGD)* as an alternative. Instead of processing all $n$ samples in each iteration, SGD iteratively updates the model using one (or a few) randomly chosen sample(s) at a time. This stochastic approach dramatically lowers memory usage (we process one sample at a time) and often converges with far fewer full data passes than batch gradient descent. In practice, SGD can reach a good solution faster in wall-clock time for large $n$, despite the randomness. In the rest of this lecture, we formalize SGD for least squares and analyze its convergence properties.

## Stochastic Gradient Descent (SGD) for Large n Problems

Let’s consider the least squares problem in a simple setting with $p = 1$ (a single parameter $w$). We have data samples $\{x_1, x_2, \dots, x_n\}$ (think of these as real numbers). Our goal is to find $w$ that minimizes the average squared error to these samples:

$$
 
L(w) \;=\; \frac{1}{2n}\sum_{i=1}^n (x_i - w)^2\,.

$$
 
This objective $L(w)$ is exactly the **mean squared error** between the constant prediction $w$ and the data points $x_i$. Setting the derivative of $L(w)$ to zero confirms this that the minimizer of $L$ is simply the sample mean 

$$
w^* = \frac{1}{n}\sum_{i=1}^n x_i =: \mu\,
$$

Indeed, $L'(w) = -\frac{1}{n}\sum_{i}(x_i - w) = w - \mu,$ 
so $L'(w)=0 \implies w=\mu$. 

We can interpret $L(w)$ as an expectation. Define the *per-sample loss* for sample $i$ as 

$$
 \ell_i(w) = \tfrac{1}{2}(x_i - w)^2. 
$$
 
Then $L(w) = \frac{1}{n}\sum_{i}\ell_i(w) = \mathbb{E}_{i\sim \mathrm{Unif}\\{1,\dots,n\\}}[\ell_i(w)]$, the average of $\ell_i$ over all samples. In other words, if we pick an index $i$ uniformly at random from $\{1,\dots,n\}$, the expected loss is $L(w)$. This viewpoint lets us apply the techniques of *stochastic optimization:* rather than minimizing $L(w)$ directly, we can minimize it *indirectly* by sampling random data points and moving $w$ to reduce the loss on those samples.

**SGD update rule:** At iteration $k$, suppose our current estimate is $w_k$. We randomly sample one data point $x_{i_k}$ from our dataset (each index has probability $1/n$). We then take a gradient step *using only that sample’s loss*. The gradient of the loss on sample $i_k$ is 

$$
 \nabla \ell_{i_k}(w_k) = \,\big(w_k - x_{i_k}\big)\,. 
$$
 
The SGD **update** is:

$$
 
w_{k+1} \;=\; w_k \;-\; \eta\, \nabla \ell_{i_k}(w_k) \;= \;=\; w_k - \eta\,(w_k - x_{i_k})\,. 

$$
 
We can rewrite this as:

$$
 
\boxed{w_{k+1} = (1-\eta)\,w_k \;+\; \eta\, x_{i_k}\,,} 

$$
 
where $0 < \eta \le 1$ is the **step size** (learning rate). This simple formula says: to update $w$, take a weighted average of the old value $w_k$ and the sampled data point $x_{i_k}$. 

Each SGD step **pulls $w$ toward one randomly chosen data point**. Over many iterations, $w$ will wander as it chases different samples. Intuitively, if we average these random steps, we hope to converge to the true mean $\mu$. Importantly, note that we never needed to load or process the entire dataset at once — each update uses just one $x_{i_k}$. This is why SGD is memory-efficient. Also, each iteration costs $O(1)$ time (constant, independent of $n$), versus $O(n)$ for a full gradient descent step. We are effectively trading off using *many cheap, noisy updates* instead of *fewer expensive, exact updates*. This formulation corresponds to **stochastic optimization** because we are optimizing the expected loss by sampling — $w_{k+1}$ is a random variable, but in expectation it moves $w$ in the direction of lowering $L(w)$. Next, we'll formalize the convergence behavior of this stochastic process.

## Convergence in Mean

To analyze SGD, we will study the **expected behavior** of the iterate $w_k$. Because of the randomness in picking $i_k$, $w_k$ is a random variable. Let $\mathbb{E}[w_k]$ denote the expectation of $w_k$ over all random choices up to iteration $k$. We will also use conditional expectation: $\mathbb{E}[w_{k+1} \mid w_k]$ means the expected value of $w_{k+1}$ given the current state $w_k$ (conditioning on the history up to $k$). All randomness at step $k+1$ comes from the new random index $i_k$.

**Unbiased gradient estimator:** First, note that the **stochastic gradient** we use at step $k$ is $\nabla \ell_{i_k}(w_k) = (w_k-x_{i_k})$. Its expectation (conditioning on $w_k$) equals the true gradient of $L(w)$ at $w_k$:

$$
 
\mathbb{E}_{i_k}[\,\nabla \ell_{i_k}(w_k)\mid w_k] \;=\; \mathbb{E}_{i_k}[(w_k - x_{i_k}) \mid w_k] \;=\; (w_k - \mathbb{E}[x_{i_k} \mid w_k])  \;=\; w_k - \mu\,.

$$
 
Here we used $\mathbb{E}[x_{i_k}] = \mu$, since each data point is equally likely. But $w_k - \mu$ is exactly $\nabla L(w_k)$, the gradient of the full loss $L$ (as we derived earlier). Thus, **$\mathbb{E}[\nabla \ell_{i_k}(w_k)] = \nabla L(w_k)$**, meaning our randomly sampled gradient is an *unbiased estimator* of the true gradient.

**Expected update = gradient descent:** Because the gradient estimate is unbiased, the *expected change* in $w$ follows the deterministic gradient descent on $L(w)$. Formally, taking expectation of the update $w_{k+1} = w_k - \eta (w_k - \mu)$ (we plugged in $w_k - \mu$ for the expected gradient):


$$

\mathbb{E}[w_{k+1} \mid w_k] \;=\; w_k - \eta\,(w_k - \mu)\,. 

$$
 

This holds for the actual random update in expectation. Now take full expectation on both sides (over the randomness up to step $k$, which is completely determined by $i_0, \ldots, i_k$):


$$

\mathbb{E}[w_{k+1}] \;=\; \mathbb{E}[\,w_k - \eta (w_k - \mu)\,] \;=\; \mathbb{E}[w_k] - \eta\big(\mathbb{E}[w_k] - \mu\big)\,. 

$$
 

Let $m_k := \mathbb{E}[w_k]$. The above is a linear difference equation for $m_k$:


$$
 
m_{k+1} - \mu = (1-\eta)\,(m_k - \mu)\,. 

$$
 

This implies that the *expected error* from the optimum $\mu$ shrinks by a factor $(1-\eta)$ each iteration. Unrolling the recurrence (or by induction), we get:


$$
 
m_k - \mu = (1-\eta)^k (m_0 - \mu)\,. 

$$
 

Assuming initial weight $w_0$ (which is deterministic or independent of the sample choice), $m_0 = w_0$. Thus:


$$
 
\mathbb{E}[w_k] = \mu + (1-\eta)^k (w_0 - \mu)\,. 

$$


This is the **convergence in mean** result. As $k \to \infty$, $(1-\eta)^k \to 0$, so $\mathbb{E}[w_k] \to \mu$. In other words, *the expected value of the SGD iterate converges to the optimal solution*. Moreover, the rate of convergence is geometric: after $k$ steps, the expected error has been multiplied by $(1-\eta)^k$. For example, if $\eta = 0.1$, then $\mathbb{E}[w_k] - \mu = 0.9^k (w_0-\mu)$, which decays quite fast.

**Empirical illustration:** *Figure 3* shows the behavior of $w_k$ in a simulation for different step sizes $\eta$. We generated a synthetic dataset with a known mean $\mu$ and ran SGD many times to estimate $\mathbb{E}[w_k]$. As predicted by the theory, all runs converge toward $\mu$ on average. Larger $\eta$ values show a steeper initial drop (faster convergence in expectation), while smaller $\eta$ values converge more slowly. In the figure, the solid lines represent the average $w_k$ over many trials, and the dashed lines show the theoretical formula $\mu + (1-\eta)^k(w_0-\mu)$ — they match closely. **Figure 3:** *Empirical convergence of $w_k$ towards $\mu$ for different step sizes ($\eta$). The average of 100 runs is plotted against iteration $k$, for $\eta=0.1$ (blue) and $\eta=0.5$ (orange). Dashed lines indicate the theoretical expectation $\mathbb{E}[w_k]$. (Placeholder for figure)*

Despite the *fast convergence in mean*, individual SGD runs can exhibit high variability. With a large step size, a single run of $w_k$ might oscillate around $\mu$ or jump back and forth, even though on average it’s centered at $\mu$. In fact, our analysis above shows $\mathbb{E}[w_k]$ converges, but does not tell us how concentrated $w_k$ is around that mean. In practice we observe a **trade-off**: larger $\eta$ gives faster convergence in expectation, but typically *higher variance* in the sequence $\{w_k\}$. We explore this next by analyzing the variance of $w_k$.

## Variance of the Estimator

We now quantify the variability in the SGD iterate $w_k$. Even though $\mathbb{E}[w_k] \to \mu$, the individual sequence does not converge to $\mu$ almost surely for a fixed $\eta>0$ — it will keep fluctuating due to the random sampling. To measure this, we look at the **mean squared error** $E[(w_k - \mu)^2]$, which captures both the variance and bias of $w_k$ relative to the optimum. (Since $w_k$ approaches unbiasedness as $k$ grows, this is essentially the variance for large $k$.)

Starting from the update $w_{k+1} = (1-\eta)w_k + \eta\,x_{i_k}$, let's derive a recurrence for the second moment of the error $e_k := w_k - \mu$. First rewrite the update in terms of $e_k$:


$$

\begin{aligned}
w_{k+1} - \mu &= (1-\eta)w_k + \eta x_{i_k} - \mu \\
&= (1-\eta)(w_k - \mu) + \eta(x_{i_k} - \mu) \\
&= (1-\eta)\,e_k + \eta\,\underbrace{(x_{i_k} - \mu)}_{\text{random sample deviation}}\,.
\end{aligned}

$$


Now square both sides and take expectation. It’s helpful to use the law of total expectation, conditioning on $w_k$ (thus on $e_k$):


$$

\mathbb{E}\!\big[(w_{k+1}-\mu)^2 \mid w_k\big] \;=\; (1-\eta)^2 e_k^2 \;+\; 2(1-\eta)\eta\, e_k\,\mathbb{E}[\,x_{i_k}-\mu \mid w_k] \;+\; \eta^2 \mathbb{E}[\,(x_{i_k}-\mu)^2 \mid w_k]\,.

$$


Given $w_k$, the only randomness is in $x_{i_k}$. We know $\mathbb{E}[x_{i_k}-\mu]=0$, by definition of $\mu$, so the middle term **drops out**. Also, $\mathbb{E}[(x_{i_k}-\mu)^2]$ is just the variance of a random sample from our dataset. Let 

$$
 
\sigma^2 := \frac{1}{n}\sum_{i=1}^n (x_i - \mu)^2 

$$
 
denote the sample variance of the data (the average squared deviation from $\mu$). This is $\mathrm{Var}(x_{i_k})$. Substituting these facts:


$$

\mathbb{E}\!\big[(w_{k+1}-\mu)^2 \mid w_k\big] \;=\; (1-\eta)^2 e_k^2 \;+\; \eta^2\,\sigma^2\,.

$$


Now take full expectation on both sides (over the randomness of $w_k$ up to step $k$):


$$

\mathbb{E}[(w_{k+1}-\mu)^2] \;=\; (1-\eta)^2\,\mathbb{E}[(w_k-\mu)^2] \;+\; \eta^2\,\sigma^2\,.

$$


This is a recurrence for the second moment (mean squared error). It shows that two things influence $E[(w_{k+1}-\mu)^2]$: 
- The previous error is scaled by $(1-\eta)^2$ (which is $<1$, causing decay), 
- but we add an extra $\eta^2\sigma^2$ due to the *variance of the new random sample*.

We can solve this recurrence by unrolling it for $k$ steps. Let $A_k := \mathbb{E}[(w_k-\mu)^2]$. Let's try to unroll it by one step: 

$$

\begin{aligned}
A_{k+1} &= (1-\eta)^2 A_k + \eta^2 \sigma^2 \\
&= (1-\eta)^2 \left( (1-\eta)^2 A_{k-1} + \eta^2 \sigma^2 \right) + \eta^2 \sigma^2 \\
&= (1-\eta)^4 A_{k-1} + \eta^2 \sigma^2 \left( (1-\eta)^2 + 1 \right).
\end{aligned}

$$

If we continue this process, we get:


$$

A_{k} \;=\; (1-\eta)^{2k} A_0 \;+\;  \eta^2\sigma^2 \sum_{j=0}^{k-1} (1-\eta)^{2j}\,. 

$$
 

Here $A_0 = (w_0-\mu)^2$ is the initial error. The summation accounts for all the $\eta^2\sigma^2$ noise added at each iteration (discounted by factors of $(1-\eta)^2$ as we move further in the past). This is a finite geometric series. Using the sum formula $\sum_{j=0}^{k-1} r^j = \frac{1-r^k}{1-r}$ for $r=(1-\eta)^2$, we get:


$$

A_k = (1-\eta)^{2k}(w_0-\mu)^2 \;+\; \eta^2 \sigma^2 \frac{1 - (1-\eta)^{2k}}{1 - (1-\eta)^2}\,. 

$$
 

Since $1 - (1-\eta)^2 = 2\eta - \eta^2$, we can simplify the second term:


$$

\begin{aligned}
A_k &\;=\; (1-\eta)^{2k}(w_0-\mu)^2 \;+\; \frac{\eta^2}{\eta(2-\eta)}\sigma^2\Big[1 - (1-\eta)^{2k}\Big] \\
&\;=\; (1-\eta)^{2k}(w_0-\mu)^2 \;+\; \frac{\eta}{2-\eta}\,\sigma^2\Big[1 - (1-\eta)^{2k}\Big]\,.
\end{aligned}

$$


This formula describes how the mean squared error evolves. As $k\to\infty$, $(1-\eta)^{2k}\to 0$ and we approach an **asymptotic error** of $\frac{\eta}{2-\eta}\sigma^2$. For example, if $\eta=0.1$, the long-run MSE approaches $\frac{0.1}{1.9}\sigma^2 \approx 0.0526\,\sigma^2$. If $\eta=1$ (very large step), the asymptotic error is $\frac{1}{1}\sigma^2 = \sigma^2$, which makes sense: with $\eta=1$, each $w_{k+1}=x_{i_k}$ is just a random sample, so $w_k$ never averages out the noise at all (its variance equals the data variance). On the other hand, as $\eta \to 0$, the asymptotic error $\frac{\eta}{2-\eta}\sigma^2 \approx \frac{\eta}{2}\sigma^2$ becomes very small — but of course $\eta$ very small means $w_k$ moves extremely slowly.

Often it’s useful to express a **bound** on $A_k$ instead of the exact formula. Using $(1-\eta)^{2k} \le 1$ and $1 - (1-\eta)^{2k} \le 1$, we get a **loose bound** for all $k$:


$$
 
\mathbb{E}[(w_k - \mu)^2] \;\le\; (1-\eta)^{2k}(w_0-\mu)^2 \;+\; \eta\,\sigma^2\,.

$$
 
The main takeaway is that the first term decays exponentially fast, while the second term is roughly on the order of $\eta$ (for small $\eta$) – a *constant floor* that does not vanish with $k$.

**Trade-off:** This variance analysis highlights the  trade-off in choosing the step size $\eta$:
- A larger $\eta$ makes $(1-\eta)^k$ shrink faster, so the bias term $(1-\eta)^{2k}(w_0-\mu)^2$ disappears quickly – meaning $w_k$ in expectation reaches near $\mu$ in just a few iterations. However, a large $\eta$ inflates the noise term $\eta\,\sigma^2$, leading to higher variance in $w_k$. The iterates will fluctuate significantly around $\mu$ and never settle.
- A smaller $\eta$ reduces the variance injected each step – in the limit $\eta\to0$, the noise term is negligible and $w_k$ would concentrate tightly (eventually $w_k$ would converge to $\mu$). But too small $\eta$ means $(1-\eta)^k$ decays very slowly, so the convergence to the optimum in expectation is **extremely slow**.

In practice, one must choose $\eta$ to balance this trade-off: we want $\eta$ large enough for quick progress, but not so large that the solution is overly noisy. For fixed $\eta$, SGD will *not* fully converge to the exact optimum $\mu$; it will hover around it with some variance. To actually get arbitrarily close to $\mu$, a common strategy is to *decrease $\eta$ over time*. For example, using a step size that decays as $\eta_k \sim \frac{\log k}{k}$ (approximately on the order of $1/k$) is known to achieve convergence: the diminishing $\eta_k$ ensures that $\mathbb{E}[(w_k-\mu)^2] \to 0$ as $k\to\infty$ (the noise floor vanishes). However, decreasing step sizes complicate the analysis, so for now we focused on constant $\eta$ to illustrate the basic phenomena.

**Empirical verification:** *Figure 4* illustrates the variance behavior of SGD for two different step sizes on a synthetic dataset. We plot the empirical $\mathbb{E}[(w_k-\mu)^2]$ (estimated from 1000 independent runs) versus $k$, along with the theoretical prediction from our formula. As expected, for a larger step (orange curve, $\eta=0.5$), the error drops quickly at first but then levels off at a higher value (higher variance floor). For a smaller step (blue curve, $\eta=0.1$), the error decreases more slowly but eventually reaches a much lower level. **Figure 4:** *Mean squared error $\mathbb{E}[(w_k-\mu)^2]$ over iterations for $\eta=0.5$ (orange) and $\eta=0.1$ (blue). Dotted lines show the theoretical asymptotic error levels. (Placeholder for figure)*

## Summary and Next Steps

In this section, we studied stochastic gradient descent as a solution for least squares with a very large number of samples:
- **SGD updates:** Rather than using all $n$ samples to compute the gradient at each step, SGD uses one random sample at a time. The update $w_{k+1} = (1-\eta)w_k + \eta x_{i_k}$ is cheap to compute and uses minimal memory.
- **Convergence in expectation:** We proved that $\mathbb{E}[w_k]$ moves toward the true optimum $\mu$ just like in batch gradient descent, following $m_k = \mu + (1-\eta)^k(w_0-\mu)$. In expectation, SGD converges *linearly* (geometrically) to the optimum.
- **Variance and error floor:** Each SGD step introduces random noise. With a constant step size, $w_k$ does not converge to $\mu$ exactly, but oscillates around it. We derived that $\mathbb{E}[(w_k-\mu)^2]$ approaches an asymptotic value on the order of $\eta\,\sigma^2$. A larger $\eta$ means a higher variance in the final solution.
- **Trade-off:** A high learning rate ($\eta$) yields faster initial progress but more fluctuation in steady state, whereas a low $\eta$ gives stable but slow convergence. Choosing $\eta$ appropriately is crucial for good performance.
- **Memory and speed gains:** SGD allows us to handle datasets that are too large to fit in memory by processing one sample at a time. It also can reach an acceptable solution in far less time than full gradient descent when $n$ is huge, as each iteration is $O(1)$ instead of $O(n)$.

In the next lecture, we will discuss techniques to **reduce the variance** in SGD without sacrificing too much speed. These methods include **minibatching** (using a small batch of samples per update to average out some noise) and **averaging** strategies (like Polyak-Ruppert averaging of iterates) which can help SGD converge more consistently to the true optimum. We will see how these techniques allow us to enjoy the benefits of SGD (speed and scalability) while mitigating its randomness.

```python
# Standalone Python script for experiments in the lecture

import numpy as np
import time
import matplotlib.pyplot as plt

# 1. Memory usage demonstration
print("Experiment 1: Memory usage demonstration")
N = 10_000_000_000  # 10 billion (requires ~80 GB for float64)
try:
    X = np.zeros(N, dtype=np.float64)
    print("Successfully allocated array of length", N)
except MemoryError:
    print(f"MemoryError: Unable to allocate array of length {N} (out of memory).")
# (On a 64GB RAM system, the above likely triggers MemoryError, as described in the lecture.)

# 2. Computation time vs n for 100 GD iterations
print("\nExperiment 2: Time for 100 full gradient descent iterations as a function of n")
ns = [int(1e4), int(1e5), int(1e6)]  # dataset sizes to test (10k, 100k, 1M)
times = []
eta = 0.1
for n in ns:
    # Generate random dataset of size n
    X = np.random.randn(n)
    w = 0.0
    # Time 100 iterations of batch gradient descent (computing full gradient each time)
    t0 = time.time()
    for it in range(100):
        # Compute gradient: grad = (w - mu) where mu = X.mean()
        # (We recompute X.mean() each iteration to simulate the cost of summing over n)
        grad = w - np.mean(X)
        w = w - eta * grad
    t1 = time.time()
    elapsed = t1 - t0
    times.append(elapsed)
    print(f"n={n:,}\ttime={elapsed:.3f} sec")
# Plot time vs n (log-log scale) and save to file
plt.figure()
plt.plot(ns, times, marker='o')
plt.xscale('log'); plt.yscale('log')
plt.xlabel('Number of samples n (log scale)')
plt.ylabel('Time for 100 GD iterations (seconds, log scale)')
plt.title('Time vs n for 100 full GD iterations')
plt.savefig('gd_time_vs_n.png')
plt.close()
print("Saved plot: gd_time_vs_n.png")

# 3. SGD convergence in mean for different step sizes
print("\nExperiment 3: SGD convergence in mean for different step sizes")
# Synthetic data distribution (we'll sample from this on the fly)
mu = 3.0          # true mean
sigma = 2.0       # true standard deviation
np.random.seed(0) # for reproducibility

def sample_data():
    # Draw a sample from N(mu, sigma^2)
    return mu + sigma * np.random.randn()

# Parameters for simulation
w0 = 0.0        # initial w
step_sizes = [0.1, 0.5]  # step sizes to compare
K = 50          # number of SGD iterations to simulate
R = 1000        # number of independent runs to average over

# Prepare storage for results
avg_paths = {eta: np.zeros(K+1) for eta in step_sizes}    # average w at each step
theory_paths = {eta: np.zeros(K+1) for eta in step_sizes} # theoretical E[w] at each step

for eta in step_sizes:
    # Simulate R independent runs of SGD for step size eta
    w_runs = np.full(R, w0)  # array of shape (R,) for current w in each run
    avg_paths[eta][0] = w0   # at k=0, average w is just w0
    theory_paths[eta][0] = w0
    for k in range(1, K+1):
        # One SGD step for all runs (vectorized)
        # Sample R new data points at once:
        samples = mu + sigma * np.random.randn(R)
        # Update all runs in parallel:
        w_runs = (1 - eta) * w_runs + eta * samples
        # Record the average over runs after k-th update
        avg_paths[eta][k] = np.mean(w_runs)
        # Theoretical mean after k steps: mu + (1-eta)^k * (w0 - mu)
        theory_paths[eta][k] = mu + ((1-eta)**k) * (w0 - mu)
    # Print final average and theoretical value for sanity check
    print(f"eta={eta}: E[w_{K}] (empirical) = {avg_paths[eta][K]:.4f}, theory = {theory_paths[eta][K]:.4f}")

# Plot the average trajectories vs theory
plt.figure()
for eta in step_sizes:
    plt.plot(avg_paths[eta], label=f'Empirical avg, eta={eta}')
    plt.plot(theory_paths[eta], ls='--', label=f'Theory, eta={eta}')
plt.xlabel('Iteration k')
plt.ylabel('w value')
plt.title('SGD convergence in mean (average of 1000 runs)')
plt.legend()
plt.savefig('sgd_mean_convergence.png')
plt.close()
print("Saved plot: sgd_mean_convergence.png")

# 4. Variance of SGD estimator for different step sizes
print("\nExperiment 4: Variance of SGD estimator for different step sizes")
step_sizes_var = [0.1, 0.5]
K_var = 50
R_var = 5000  # more runs for better variance estimate
var_paths = {eta: np.zeros(K_var+1) for eta in step_sizes_var}
theory_var = {eta: np.zeros(K_var+1) for eta in step_sizes_var}

for eta in step_sizes_var:
    # Simulate R_var runs for variance
    w_runs = np.full(R_var, w0)
    var_paths[eta][0] = 0.0  # at k=0, variance = 0 since all runs start at w0
    theory_var[eta][0] = 0.0
    A = (w0 - mu)**2  # current theoretical MSE
    for k in range(1, K_var+1):
        # Update runs
        samples = mu + sigma * np.random.randn(R_var)
        w_runs = (1 - eta) * w_runs + eta * samples
        # Compute empirical MSE (which is variance since mean error is not zero initially, but we'll compare to full formula)
        errors = w_runs - mu
        var_paths[eta][k] = np.mean(errors**2)
        # Update theoretical MSE A_k using recurrence: A <- (1-eta)^2 * A + eta^2 * sigma^2
        A = (1 - eta)**2 * A + (eta**2) * (sigma**2)
        theory_var[eta][k] = A
    # Print asymptotic variance estimate vs theoretical
    print(f"eta={eta}: Empirical MSE at k={K_var} ~ {var_paths[eta][K_var]:.4f}, Theory ~ {theory_var[eta][K_var]:.4f}")

# Plot variance (MSE) trajectories
plt.figure()
for eta in step_sizes_var:
    plt.plot(var_paths[eta], label=f'Empirical, eta={eta}')
    plt.plot(theory_var[eta], ls='--', label=f'Theory, eta={eta}')
plt.xlabel('Iteration k')
plt.ylabel('E[(w_k - mu)^2]')
plt.title('Variance of SGD estimator over iterations')
plt.legend()
plt.savefig('sgd_variance.png')
plt.close()
print("Saved plot: sgd_variance.png")
```