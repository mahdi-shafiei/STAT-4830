---
layout: course_page
title: Stochastic Gradient Descent - Insights from a Noisy Quadratic Model
---

# Stochastic Gradient Descent in a Noisy Quadratic Model

## Introduction

Stochastic Gradient Descent (SGD) is the workhorse for large-scale machine learning, but in high dimensions its performance can suffer due to **ill-conditioning** and **gradient noise**. *Ill-conditioning* means the Hessian of the loss has a large **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ (ratio of largest to smallest eigenvalue). This causes a gap in per-coordinate convergence speeds: steep directions (large curvature $\lambda$) and flat directions (small $\lambda$) progress at very different rates under a single learning rate. *Gradient noise* refers to the variance in stochastic gradients—SGD uses noisy gradient estimates which can have different variance in different directions (described by a covariance matrix $C$). Both factors are common in deep learning: neural network losses often have extremely large $\kappa$ and stochastic gradients with high variance. As a result, vanilla SGD may converge slowly in some directions and plateau at a noise-induced floor.

**Why modify SGD?** In deep learning practice, several modifications to SGD are crucial for faster convergence:
- **Momentum:** Accelerates SGD by accumulating a velocity vector (an exponential average of past gradients) to damp oscillations and push through plateaus.
- **Exponential Moving Averages (EMA):** A form of averaging either the gradients or the model parameters to reduce variance. For instance, Polyak averaging or EMA of weights can yield more stable convergence by smoothing out noise.
- **Preconditioning:** Rescales the gradient using an approximate inverse Hessian or adaptive coordinate-wise steps. Optimizers like **Adam** and **K-FAC** implement preconditioning by adjusting learning rates per parameter (Adam uses estimates of second moments, K-FAC uses an approximation to the Fisher/Hessian). Preconditioning tackles ill-conditioning by effectively reducing $\kappa$.
- Learning rate schedules (decay) are another tool, but here we focus on the above techniques which modify the update rule itself.

In large-scale training, these techniques have been observed to impact how well we can utilize parallelism via larger batches. Empirical studies ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=size%20on%20neural%20network%20training,batch%20size%20depends%20on%20the)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=provided%20experimental%20evidence%20that%20the,2018%5D%20are%20essential%20for)) found that increasing the batch size $B$ gives near-perfect speedup (half the steps for double $B$) up to a **critical batch size**, beyond which returns diminish. Interestingly, the critical batch size depends on the optimizer: momentum extends the region of perfect scaling compared to plain SGD, and adaptive methods like Adam or K-FAC extend it even further ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). In other words, algorithms with better preconditioning can make use of much larger batches before hitting the point of diminishing returns ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). This motivates studying *why* momentum and preconditioning help, and how gradient noise and curvature interplay.

**A simple model:** Zhang et al. (2019) propose a *Noisy Quadratic Model (NQM)* ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)) as a minimal theoretical playground to analyze these questions. The NQM is a stochastic optimization problem where the true loss surface is a quadratic bowl, and the stochastic gradients are corrupted by noise. Despite its simplicity, this model manages to **capture many behaviors of neural network training** (learning rate scaling, critical batch effects, momentum/preconditioning benefits, etc.) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). Because it is analytically tractable, we can derive explicit formulas for convergence rates and noise levels, helping us understand how each SGD modification works. In the following, we use the NQM to study SGD dynamics in high dimensions and to demonstrate the effects of momentum, EMA, and preconditioning.

## Problem Setup: Noisy Quadratic Model

Consider a $d$-dimensional quadratic loss function with additive gradient noise. We assume the true (population) loss is a convex quadratic: 


$$
L(\theta) \;=\; \frac{1}{2}\,\theta^\top H\,\theta,
$$
 

where $H$ is a fixed $d\times d$ positive-definite Hessian matrix. Without loss of generality (by rotating coordinates), we assume $H$ is diagonal with entries $h_1,\dots,h_d$ on the diagonal ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=We%20now%20introduce%20the%20noisy,%E2%9C%93%29%20%3D%201)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=i%3D1%20hi%E2%9C%932%20i%20%2C%20X,To%20reduce%20the)). We also assume the optimum is at $\theta^* = 0$ (since we can center the problem at the optimum) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=We%20now%20introduce%20the%20noisy,%E2%9C%93%29%20%3D%201)). Thus each coordinate $\theta_i$ corresponds to an independent quadratic well $L_i(\theta_i) = \frac{1}{2}h_i\,\theta_i^2$. We order the coordinates so that $h_1 \ge h_2 \ge \cdots \ge h_d > 0$; here $h_1$ is the largest curvature and $h_d$ the smallest. The **condition number** of this quadratic is $\kappa = h_1/h_d$. In high dimensions, $\kappa$ can be very large (Zhang et al. set $h_i = 1/i$ in some experiments, giving $\kappa \approx d$ of order $10^4$ for $d=10^4$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=considerations%20about%20neural%20net%20training,Ubaru%20et%20al)), resembling the heavy-tailed spectrum of neural network Hessians).

Now, *stochastic* gradients come from a noisy oracle: at any point $\theta$, instead of the true gradient $H\theta$, we observe 


$$
g(\theta) \;=\; H\,\theta \;+\; \varepsilon,
$$
 

where $\varepsilon$ is a random noise vector with mean $0$ and covariance $\Cov(\varepsilon) = C$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Without%20loss%20of%20generality%2C%20we,batch)). We assume $H$ and $C$ are simultaneously diagonalizable (they share eigenvectors) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=the%20covariance%20Cov,each%20dimension%20evolves%20independently%20as)); since $H$ is diagonal in our coordinate basis, this means $C = \diag(c_1,\dots,c_d)$ is also diagonal in the same basis ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=the%20covariance%20Cov,each%20dimension%20evolves%20independently%20as)). The value $c_i$ represents the variance of the noise in the $i$th coordinate (for one sample). Intuitively, $c_i$ quantifies how “noisy” the gradient is in direction $i$. This setup mimics SGD on a least-squares problem: $H$ is like a true Hessian, and $\varepsilon$ is the difference between a stochastic gradient (from a randomly sampled data point or mini-batch) and the true gradient. **Mini-batch of size $B$:** If we average $B$ independent samples to compute a gradient, the noise variance scales down by $B$. Thus, the covariance of a mini-batch gradient is $C/B$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Without%20loss%20of%20generality%2C%20we,batch)). In particular, the noise in coordinate $i$ with batch $B$ has variance $c_i/B$. Larger $B$ means a less noisy gradient estimate.

For concreteness, one special case is worth noting: $C = H$. This means $c_i = h_i$ for all $i$, i.e. noise variance is proportional to curvature. This situation arises under certain assumptions in neural network training (when the model is near the optimum and the output distribution matches the target distribution) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=We%20also%20set%20C%20%3D,and%20moderately%20well%20for%20a)), and has been observed empirically in some cases ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=We%20also%20set%20C%20%3D,and%20moderately%20well%20for%20a)). We will not assume $C=H$ in derivations (we keep general $c_i$), but we’ll see this case often in examples since it’s analytically convenient and roughly reflects practical scenarios.

**SGD update equation:** Starting from an initial parameter $\theta(0)$, **Stochastic Gradient Descent** iteratively updates


$$
\theta(t+1) \;=\; \theta(t)\;-\;\alpha\,g(\theta(t)),
$$


where $\alpha>0$ is the learning rate. Substituting $g(\theta) = H\theta + \varepsilon$, we get the linear stochastic recurrence:


$$

\theta(t+1) \;=\; \theta(t)\;-\;\alpha\,[H\,\theta(t) + \varepsilon(t)] 
\;=\; (I - \alpha H)\,\theta(t)\;-\;\alpha\,\varepsilon(t).

$$


In coordinate form (for $i=1,\dots,d$):


$$

\theta_i(t+1) \;=\; (1 - \alpha\,h_i)\,\theta_i(t)\;-\;\alpha\,\varepsilon_i(t),

$$
 

with $\varepsilon_i(t)\sim \mathcal{N}(0,c_i)$ (or more generally $\Var(\varepsilon_i)=c_i$) for the *batch-$1$ case ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=%E2%9C%93i,a%20given%20dimension%20i%20is)). If a mini-batch of size $B$ is used, $\varepsilon_i(t)$ has variance $c_i/B$ but the form is the same ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Without%20loss%20of%20generality%2C%20we,batch)).

This update is a linear system, so we can analyze each coordinate independently. Let $r_i = 1 - \alpha h_i$ be the contraction factor for coordinate $i$. We assume the step size is small enough to ensure $|r_i|<1$ for all $i$ (in fact, for convex quadratics it suffices to assume $\alpha h_i < 2$ for stability ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=%2C%20,2014)); typically $\alpha \le 2/h_1$). The *expected dynamics* per coordinate are easy to derive:
- **Mean:** Taking expectation in the update, $E[\varepsilon_i(t)]=0$, so 
  
$$

  E[\theta_i(t+1)] \;=\; (1-\alpha h_i)\,E[\theta_i(t)]. 
  
$$
 
  Thus $E[\theta_i(t)] = (1-\alpha h_i)^t\,\theta_i(0)$. Each component’s mean decays geometrically to $0$. The larger $h_i$ is, the faster the mean decays (since $|1-\alpha h_i|$ is smaller).
- **Variance:** Squaring the update and using independence of $\theta(t)$ and $\varepsilon(t)$, one can show 
  
$$

  \Var[\theta_i(t+1)] \;=\; (1-\alpha h_i)^2\,\Var[\theta_i(t)] \;+\; \alpha^2\,\frac{c_i}{B}\,. 
  
$$
 
  Intuitively, whatever variance existed at time $t$ is multiplied by $(1-\alpha h_i)^2$ (because the deterministic part scales the variable), and new variance $\alpha^2 c_i/B$ is added due to the fresh noise at step $t$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=%E2%9C%93i,a%20given%20dimension%20i%20is)).

From these, we can understand the convergence of $\theta_i$. If there were no noise ($c_i=0$), $\theta_i(t)$ would converge to 0 at a rate $(1-\alpha h_i)^t$; in particular, the *error* decays exponentially with rate constant $1-\alpha h_i$. With noise, however, variance keeps getting injected, so $\theta_i(t)$ will hover around $0$ in steady-state. In fact, as $t\to\infty}$, $\Var(\theta_i(t))$ approaches a finite limit determined by balancing decay and injection:

$$

\Var[\theta_i(\infty)] \;=\; \frac{\alpha^2 (c_i/B)}{1 - (1-\alpha h_i)^2}\,.

$$

The *mean squared* value of $\theta_i$ thus converges to a nonzero limit. If we consider the **population loss** in that coordinate, $\ell_i(t) = \frac{1}{2}h_i\,\theta_i(t)^2$, its expectation tends to a *steady-state risk*:

$$

E[\ell_i(\infty)] \;=\; \frac{1}{2}h_i\,\Var[\theta_i(\infty)] \;=\; \frac{\alpha\,c_i}{2B\,[2 - \alpha h_i]}\,. 

$$
 
This is the irreducible error due to noise. The **population risk** (total loss) is $E[L(\theta(t))] = \sum_{i=1}^d E[\ell_i(t)]$, which will approach $\sum_i \frac{\alpha c_i}{2B(2-\alpha h_i)}$ as $t\to\infty$.

Fortunately, before reaching that floor, SGD does move toward the optimum. We can solve the recursion exactly to get the expected loss at any finite time. Starting from some initial $\theta(0)$, one can show ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Based%20on%20eqn,4)):


$$

E[\ell_i(t)] \;=\; (1-\alpha h_i)^{2t}\;E[\ell_i(0)] \;+\;\Big(1 - (1-\alpha h_i)^{2t}\Big)\,\frac{\alpha\,c_i}{2B\,(2-\alpha h_i)}\,.

$$


This formula has two terms: **(i)** a decaying term coming from the initial error, which shrinks as $(1-\alpha h_i)^{2t}$, and **(ii)** an asymptotic term that represents the noise floor (steady-state risk). We see that each coordinate’s loss converges exponentially fast to $\frac{\alpha c_i}{2B(2-\alpha h_i)}$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Based%20on%20eqn,4)). The convergence rate is governed by $1-\alpha h_i$ (faster for larger $h_i$), while the limiting error is larger for larger $c_i$ and smaller $B$ (and also depends on $h_i$ via $2-\alpha h_i$ in the denominator).

**Key observation:** There is a trade-off between convergence speed and final accuracy:
- A larger learning rate $\alpha$ (closer to the stability limit $2/h_i$) makes $(1-\alpha h_i)$ smaller in magnitude, so the initial error decays **faster** (exponential convergence with a larger rate constant). However, a larger $\alpha$ also *raises the noise floor* $\frac{\alpha c_i}{2B(2-\alpha h_i)}$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Remarkably%2C%20each%20dimension%20converges%20exponentially,momentum%20and%20preconditioning%29%20help)), resulting in a higher steady-state error.
- Increasing the batch size $B$ *lowers the noise floor* (steady-state risk) proportionally, since variance injection $\sim 1/B$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=a%20trade,increasing%20the%20steady%20state%20risk)). However, $B$ does not affect the convergence rate $(1-\alpha h_i)$ for a given $\alpha$ (it only reduces the noise). In practice, this means larger batches can achieve lower final loss, but if we do not change $\alpha$, they won’t by themselves speed up the initial convergence per iteration (they do allow using a larger $\alpha$ without divergence, which *can* speed up convergence – more on that later).

Now, because $h_1 \ge h_2 \ge \cdots \ge h_d$, for a given fixed $\alpha$, different coordinates converge at different rates. The largest-eigenvalue direction $i=1$ has the factor $|1-\alpha h_1|$ – this might be small (fast decay) if we choose $\alpha$ close to $1/h_1$. Meanwhile, a direction with a much smaller curvature $h_d$ decays with factor $1-\alpha h_d \approx 1-\alpha h_1 \cdot (h_d/h_1)$. If $\kappa = h_1/h_d$ is large, $1-\alpha h_d \approx 1 - (\alpha h_1)/\kappa$. For stability we require $\alpha h_1 \lesssim 2$, so $1-\alpha h_d \gtrsim 1 - 2/\kappa$. For large $\kappa$, this is near $1$, meaning *very slow decay*. In short, **ill-conditioning causes a huge disparity in convergence rates across coordinates**. The slowest direction (smallest $h_i$) will dominate the overall convergence. The *total loss* $E[L(\theta(t))] = \sum_i E[\ell_i(t)]$ will be initially dominated by the slowest-decaying term (assuming we start with comparable initial error in each direction). This effect is illustrated conceptually in Fig. 1 of Zhang et al. (2019): without any adjustment, the loss in the low-curvature directions persists much longer, causing a long tail in the training curve ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=decay%20at%20steS%20100%20Figure,we%20assume%20without%20loss%20of)).

To make matters worse, the slowest direction also has the *highest relative noise* effect if $c_i$ does not decrease with $h_i$. In the worst case $c_i$ might even be larger for smaller $h_i$ (though if $C$ is aligned with $H$, one often expects $c_i$ and $h_i$ to be positively correlated ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=We%20also%20set%20C%20%3D,and%20moderately%20well%20for%20a))). Regardless, when $\alpha$ is tuned to the largest curvature for stability, the *effective* noise-to-signal ratio is typically worst in the flat directions.

**Implications:** Plain SGD in a high-$\kappa$ problem will converge *slowly*, essentially limited by the hardest (smallest $h$) direction, and will settle at an error level $\approx \frac{\alpha}{2B} \sum_i \frac{c_i}{2-\alpha h_i}$ (which in worst case scales with $\frac{\alpha}{B}\sum_i c_i$). Two obvious ways to improve performance are:
1. **Reduce the noise variance** $c_i/B$ by increasing batch size $B$. This lowers steady-state error and allows using a larger $\alpha$ (since larger batches make the optimization more deterministic). A larger $\alpha$ in turn speeds up convergence in all directions. This is why we often scale learning rate up with batch size in practice. However, beyond some point, simply increasing $B$ yields diminishing returns ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=size%20on%20neural%20network%20training,batch%20size%20depends%20on%20the)): eventually the noise floor is low enough that error is dominated by the *bias* (initial error that hasn’t decayed yet) rather than variance, or we hit the maximum stable learning rate. The **critical batch size** is roughly when further reduction in noise doesn’t significantly reduce training steps needed ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=size%20on%20neural%20network%20training,batch%20size%20depends%20on%20the)).
2. **Speed up the slow directions without blowing up the fast ones** – this is exactly what momentum and preconditioning aim to do. Momentum tries to accelerate convergence (especially for those slow directions) without simply increasing $\alpha$, and preconditioning explicitly rescales each direction to shrink $\kappa$. These methods *may increase the noise floor*, but if combined with larger $B$, their full benefit can be realized ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=proportionally%20to%20increases%20in%20batch,as%20shown%20in%20later%20sections)).

In the following sections, we examine how **Momentum**, **Exponential Moving Average (EMA)**, and **Preconditioning** modify the SGD dynamics in the NQM. We will derive their update rules and see their effect on convergence rates and steady-state variance. We will also use small Python simulations of the NQM (2D examples) to illustrate these effects. Finally, we connect these insights to deep learning by summarizing findings from Zhang *et al.* (2019).

## Understanding SGD Dynamics via the NQM

Before adding fancy modifications, let's deepen our understanding of plain SGD on the noisy quadratic with a concrete example and some visualizations. We will consider a 2D case ($d=2$) so that we can plot the error in two orthogonal directions.

**Example setup:** Let $h_1 = 50$, $h_2 = 1$ (condition number $\kappa = 50$). We take the noise variances proportional to the Hessian ($c_1 = 50$, $c_2 = 1$), which means noise is higher in the stiff (high curvature) direction – this roughly mimics the idea that gradients in steep directions have larger variance ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=We%20also%20set%20C%20%3D,and%20moderately%20well%20for%20a)). Suppose we start at $\theta(0) = (1,\,1)$, so initial errors in both coordinates are equal. We choose a learning rate $\alpha = 0.02$, which is  somewhat aggressive (for $h_1=50$, $\alpha h_1 = 1$ which is 50% of the stability limit $2/h_1$).

Let's simulate SGD for this scenario and compare with the theory:

```python
import numpy as np

# Problem parameters
h1, h2 = 50.0, 1.0
c1, c2 = 50.0, 1.0   # noise variances
H = np.array([h1, h2])
C = np.array([c1, c2])

alpha = 0.02
B = 1        # batch size
steps = 200
trials = 10000  # number of independent trials for averaging

rng = np.random.default_rng(seed=42)
# Initialize theta for all trials
theta = np.tile([1.0, 1.0], (trials, 1))  # each trial starts at (1,1)

# Run SGD simulation
for t in range(steps):
    noise = rng.normal(0, 1, size=(trials, 2)) * np.sqrt(C/B)  # noise ~ N(0, C/B)
    grad = theta * H + noise      # stochastic grad = H*theta + noise
    theta = theta - alpha * grad  # SGD update

# Compute the expected squared theta values from simulation
mean_theta_sq = (theta**2).mean(axis=0)
print("Estimated E[θ_1^2] =", mean_theta_sq[0])
print("Estimated E[θ_2^2] =", mean_theta_sq[1])
```

Running the above (with a large number of trials to average out noise) yields an estimate of $E[\theta_1^2(t)]$ and $E[\theta_2^2(t)]$ at $t=200$. We can compare these to the theoretical formula. According to our earlier analysis, for each coordinate $i$:


$$
E[\theta_i^2(t)] \;=\; (1-\alpha h_i)^{2t}\,\theta_i(0)^2 \;+\; \frac{\alpha^2 c_i/B}{1 - (1-\alpha h_i)^2}\Big(1 - (1-\alpha h_i)^{2t}\Big).
$$


Let's compute the theoretical values for $t=200$:

```python
# Theoretical values
r1 = 1 - alpha * h1
r2 = 1 - alpha * h2
theta0 = np.array([1.0, 1.0])
E_theta1_sq = (r1**(2*steps)) * theta0[0]**2 + (alpha**2 * c1 / B) * (1 - r1**(2*steps)) / (1 - r1**2)
E_theta2_sq = (r2**(2*steps)) * theta0[1]**2 + (alpha**2 * c2 / B) * (1 - r2**(2*steps)) / (1 - r2**2)
print("Theoretical E[θ_1^2] =", E_theta1_sq)
print("Theoretical E[θ_2^2] =", E_theta2_sq)
```

Both the simulation and theory should roughly agree (up to sampling error). We expect $\theta_1$ (the stiff direction) to have decayed close to its noise floor, and $\theta_2$ (the flat direction) to still be decaying. Let's interpret the numbers:

- **Fast direction (1)**: $h_1=50$. With $\alpha=0.02$, $1-\alpha h_1 = 1 - 1 = 0$. So in theory $\theta_1$'s mean should drop to essentially $0$ in one step! Indeed, after the first iteration the mean of $\theta_1$ becomes zero (we hit the minimum along that direction, since we slightly underdamped it). After that, $\theta_1$ just fluctuates due to noise. Its variance should reach $\approx \frac{\alpha^2 c_1}{1-(1-\alpha h_1)^2} = \frac{0.0004 \cdot 50}{1-0^2} = 0.02$ eventually, so $E[\theta_1^2]$ should approach $0.02$. Our simulation (and formula) at $t=200$ will show $E[\theta_1^2] \approx 0.02$, meaning $\theta_1 \approx 0$ on average but $\text{Std}(\theta_1)\approx \sqrt{0.02}\approx 0.14$ due to noise. This corresponds to a steady-state contribution to loss $\approx \frac{1}{2}h_1 E[\theta_1^2] = 25 \times 0.02 = 0.5$. So the first coordinate contributes about $0.5$ to the loss floor.
- **Slow direction (2)**: $h_2=1$. Here $1-\alpha h_2 = 0.98$. So the mean decays as $(0.98)^t$. After $200$ steps, $(0.98)^{200} \approx 0.0176$, so the mean of $\theta_2$ is about $0.0176$ (starting from $1$). The variance injection per step is very small since $\alpha^2 c_2 = 0.0004$, and $(1-0.98^2) = (1-0.9604)=0.0396$ in the denominator, so eventual $\Var(\theta_2) \approx 0.0101$. At $t=200$, it will be slightly less. The simulation might show $E[\theta_2^2] \approx 0.0104$, composed of $(E[\theta_2])^2 \approx (0.0176)^2 = 0.00031$ and $\Var(\theta_2)\approx 0.0101$. The loss from coordinate 2 is about $\frac{1}{2} \cdot 1 \cdot E[\theta_2^2] \approx 0.005$. So the flat direction contributes very little to loss at this point (because its $\theta$ value is small *and* its curvature is low).

Summing them up, the expected total loss $E[L(\theta(200))] \approx 0.5 + 0.005 = 0.505$. This is essentially the steady-state loss already – the first direction hit its noise floor of $0.5$ very quickly, and the second is still decaying but only contributes a tiny amount anyway. With more iterations, $\theta_2$ will eventually also settle at its floor (which is $\frac{\alpha c_2}{2(2-\alpha)} \approx 0.0051$). So the final loss will be about $0.5051$.

In summary, in this example the **stiff direction** (high curvature) converged fast but to a relatively high variance (since noise in that direction is large), whereas the **flat direction** is exceedingly slow but will eventually reach a much lower variance. This shows the dilemma: if we tune $\alpha$ for the highest curvature, we live with a large noise floor from that coordinate; if we tune $\alpha$ lower, we'd reduce that noise but slow down overall progress even more. Simply increasing batch size $B$ would lower both noise floors proportionally – e.g. if we used $B=50$ in the above, the loss floor would drop by $50\times$ to $\sim0.01$ (and we could possibly double $\alpha$ safely). But large $B$ means more computation per step. Techniques like momentum, EMA, and preconditioning seek *algorithmic* solutions to get the best of both worlds: fast convergence and low error, without always requiring huge batch sizes.

We will now discuss each technique in turn, analyzing how they modify the SGD update and what effect they have on the dynamics of each coordinate.

## Exponential Moving Averages (EMA)

One simple way to reduce the variance of SGD is to **average the iterates**. Instead of using the last point $\theta(t)$ as our final answer, we can keep a running average of the parameters over time, which smooths out the random fluctuations. In online learning theory, this idea is known as **Polyak-Ruppert averaging**: the average of all past SGD iterates often converges faster (in mean squared error) than the raw iterate. In practice, a convenient method is to maintain an **exponential moving average (EMA)** of the parameters:

- Initialize $\tilde{\theta}(0) = \theta(0)$.
- At each step $t$, after updating $\theta(t)$, update the EMA:
  
$$

  \tilde{\theta}(t) \;=\; \beta\,\tilde{\theta}(t-1) \;+\; (1-\beta)\,\theta(t),
  
$$
 
  where $\beta \in [0,1)$ is a decay factor close to 1 (e.g. $\beta=0.9$ or $0.99$). This can be rewritten as $\tilde{\theta}(t) = \tilde{\theta}(t-1) + (1-\beta)[\theta(t) - \tilde{\theta}(t-1)]$, showing that $\tilde{\theta}$ is a low-pass filtered version of $\theta$. Only a $(1-\beta)$ fraction of the new update is incorporated each time.

Equivalently, $\tilde{\theta}(t)$ is the weighted average of all past $\theta$ values with exponentially decaying weights (most weight on recent ones). If $\beta$ is close to 1, the average is long-term (heavy smoothing); if $\beta$ is lower, the average emphasizes more recent iterations. In the special case $\beta=0$, $\tilde{\theta}(t)$ just equals the current $\theta(t)$ (no averaging). As $\beta \to 1$, the EMA approaches the simple average of all iterates.

Importantly, this EMA does **not** affect the training dynamics of $\theta(t)$ itself – it is a **post-processing** technique. We still update $\theta(t)$ by SGD as usual, but we keep track of $\tilde{\theta}(t)$ in parallel and consider $\tilde{\theta}(t)$ as our estimator of the optimum. The overhead is minimal: just storing one extra vector and an $O(d)$ update each iteration.

**Variance reduction effect:** Intuitively, $\tilde{\theta}(t)$ has lower variance than $\theta(t)$. Each new $\theta(t)$ has noise, but averaging multiple iterates will cancel out some of this noise. If the noise is roughly i.i.d. over time (not exactly true here, but let’s assume the dominant noise is from gradients which are mostly uncorrelated over iterations), averaging $N$ samples reduces variance by about $N$-fold. EMA with $\beta$ close to 1 behaves like an average over roughly $\frac{1}{1-\beta}$ recent samples. For example, $\beta=0.9$ effectively averages about 10 points, $\beta=0.99$ averages about 100 points. We might thus expect an EMA to reduce the steady-state variance (and risk) by a factor on the order of $\sim (1-\beta)/(1+\beta)$ or similar (the exact factor requires solving the linear system). Crucially, averaging does **not** bias the result – $\E[\tilde{\theta}(t)] = E[\theta(t)]$ (the average remains centered on the true optimum, assuming $\E[\theta(t)] = \theta^*$). Thus, EMA can reduce *error variance* without hurting the convergence of the mean.

Indeed, a rigorous analysis of EMA on the NQM shows that we can tune $\beta$ to achieve **strictly lower steady-state risk** than plain SGD, with no slowdown in convergence rate ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=By%20properly%20choosing%20an%20averaging,5)). In particular, if we choose $\beta$ such that $\beta < 1 - \alpha h_d$ (so that the EMA’s memory decays slightly faster than the slowest mode of $\theta$), then the convergence rate is still dominated by $(1-\alpha h_d)$ (essentially unchanged from plain SGD), while the variance in each coordinate is reduced by a factor involving $(1-\beta)$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=By%20properly%20choosing%20an%20averaging,5)). The exact formula from Zhang et al. (2019) is a bit complicated, but they highlight that the steady-state risk term is multiplied by a factor 

$$
 \frac{(1-\beta)(1 + (1-\alpha h_i))}{(1+\beta)(1 - (1-\alpha h_i))} < 1, 
$$
 
which is strictly less than 1 (for $0<\beta<1$) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=By%20properly%20choosing%20an%20averaging,5)). Thus **EMA reduces the noise floor** in all coordinates.

To illustrate, let's revisit the earlier example (with $h_1=50, h_2=1, \alpha=0.02, B=1$) and apply an EMA to the SGD iterates. We choose, say, $\beta=0.9$ (which corresponds to roughly a 10-step half-life in the average). We will simulate SGD and compute the mean squared error of the EMA estimator versus the raw parameter.

```python
# Simulate SGD with EMA for the same 2D example
alpha = 0.02
beta = 0.9
steps = 2000  # run longer to reach steady-state
trials = 1000

rng = np.random.default_rng(seed=123)
theta = np.tile([1.0, 1.0], (trials, 1))
theta_bar = np.tile([1.0, 1.0], (trials, 1))  # initialize EMA equal to theta

# Run SGD with EMA
for t in range(steps):
    noise = rng.normal(0, 1, size=(trials, 2)) * np.sqrt(C/B)
    grad = theta * H + noise
    theta = theta - alpha * grad
    # update EMA of theta
    theta_bar = beta * theta_bar + (1-beta) * theta

# Compute mean squared norm of error for theta and theta_bar
# (Since optimum is 0, this is just E[||theta||^2] = E[theta_1^2] + E[theta_2^2])
mse_theta = (theta**2).sum(axis=1).mean()
mse_theta_bar = (theta_bar**2).sum(axis=1).mean()
print("Mean squared error of raw θ:", mse_theta)
print("Mean squared error of EMA θ̄:", mse_theta_bar)
```

We expect $\tilde{\theta}$ (EMA) to have lower MSE than $\theta$. In fact, Zhang et al. report that EMA can achieve the **same level of error with a smaller batch size**: *“EMA reduces the number of steps required... and becomes redundant in the large batch (near-deterministic) regime since increasing the batch size can also reduce the variance”* ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)). In other words, EMA effectively shifts the critical batch size to a smaller value — you can get away with a smaller batch to achieve a given error if you use EMA. This is because averaging mimics the effect of having more samples per update. Their experiments confirmed that with EMA, the training curve reaches a target loss in fewer steps (especially for small batches) compared to without EMA ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)). At very large $B$, EMA offers little benefit because the gradients are nearly deterministic (noise is low) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)).

In summary, **Exponential Moving Average is a pure variance-reduction technique**. It does not change the expected trajectory of SGD (the bias), but it smooths out the fluctuations:
- It has **no effect on the convergence rate** of the mean if tuned properly (aside from a negligible startup transient).
- It **significantly lowers the steady-state variance**. In our example, we would see that $E[\|\tilde{\theta}\|^2] < E[\|\theta\|^2]$ by a noticeable factor.
- It is most useful in the regime where noise is limiting performance (small batches or late in training). In very large-batch or very late-phase training, EMA’s benefit diminishes, as noted in the NQM analysis and experiments ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)).
- EMA is simple to implement and often used in practice for obtaining a better final model (e.g. in reinforcement learning or generative models, where an EMA of the weights is saved for inference). 

**Connection to deep learning:** The theoretical analysis predicts that EMA provides variance reduction without slowing training. Empirically, Zhang et al. found that using EMA in training (of CNNs on CIFAR10/MNIST) reduced the steps needed at small batch sizes, and that without EMA one would need a larger batch to achieve the same error ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)). This aligns with our understanding that averaging imitates having more samples. In practice, one might either use EMA or simply increase batch size (or train longer) to reduce variance – EMA gives a cheaper alternative in some cases. It’s also worth noting that EMA of model weights is sometimes used to improve *generalization* (it smooths out the noise in parameter space, often leading to lower validation loss even if training loss is similar).

## Momentum

**Momentum SGD** refers to SGD equipped with a momentum term, often implemented as the *heavy-ball* method. The update with momentum introduces a velocity variable $v(t)$ that accumulates past gradients:


$$

v(t+1) = \beta\, v(t) + g(\theta(t+1)),

$$
 

$$

\theta(t+1) = \theta(t) - \alpha\, v(t+1),

$$
 

where $\beta \in [0,1)$ is the momentum coefficient (sometimes denoted by $\mu$). In words, at each step we take the new gradient and add it to a scaled version of the previous velocity. Equivalently, we can write the combined update as:

$$

\theta(t+1) = \theta(t) - \alpha\,[\underbrace{H \theta(t) + \varepsilon(t)}_{\text{current grad}}] + \beta\,[\theta(t) - \theta(t-1)],

$$
 
which shows that momentum adds a fraction $\beta$ of the last step $\Delta \theta(t) = \theta(t) - \theta(t-1)$ to the new step. This has the effect of “smoothing” the direction of travel – gradients from several consecutive steps are aggregated to determine the update direction. Momentum can help accelerate descent in directions where gradients are consistent and damp oscillations in directions where they change sign (e.g. bouncing in narrow ravines).

In the case of the NQM, where everything is linear-quadratic, we can analyze momentum by looking at each coordinate $i$. The update for coordinate $i$ with momentum is:

$$

m_i(t+1) = \beta\, m_i(t) + h_i\,\theta_i(t) + \varepsilon_i(t),

$$
 

$$

\theta_i(t+1) = \theta_i(t) - \alpha\, m_i(t+1),

$$
 
where $m_i$ is the momentum buffer for coordinate $i$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=3,batch%20regime%2C%20which)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=momentum%20SGD%20are%3A%20mi,that%20momentum%20SGD)). (We obtained this by expanding $v$ and noting $g(\theta)=h_i \theta_i + \varepsilon_i$.) This is a second-order linear difference equation in $\theta_i$. One can solve it or find its characteristic polynomial:

$$
r^2 - (1 - \alpha h_i + \beta)\,r + \beta = 0.
$$
 
The roots of this equation (call them $r_{1,2}$) determine the behavior. If the roots are real, we have an overdamped system; if complex (conjugates), we have an underdamped (oscillatory) system. The momentum parameter $\beta$ basically pushes the system toward the oscillatory regime if it’s too high. Specifically, the boundary between underdamped and overdamped occurs when the discriminant $(1-\alpha h_i + \beta)^2 - 4\beta$ goes to zero, i.e. $\beta = (1-\alpha h_i)^2$ (this is when the two roots merge) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=,7)). For a given $\alpha h_i$, if $\beta$ is larger than $(1-\alpha h_i)^2$, the roots become complex and you get oscillations. For example, if $\alpha h_i$ is small, $(1-\alpha h_i)^2 \approx 1$, so only $\beta$ extremely close to 1 will cause oscillations. But if $\alpha h_i$ is moderately large, a smaller $\beta$ can still cause underdamping.

Zhang et al. suggest choosing $\beta$ such that *all* dimensions are in the overdamped regime (to avoid oscillations in the slowest direction) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=In%20the%20case%20of%20underdamping,the%20overdamping%20regime%20as%20argued)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=simply%20replacing%20the%20learning%20rate,Therefore%2C%20we)). That means $\beta \le (1 - \alpha h_d)^2$ (where $h_d$ is smallest eigenvalue). In practice, one often uses $\beta=0.9$ or $0.99$ which usually yield a well-behaved trajectory if learning rates are not too large.

**Impact on convergence:** The momentum update effectively introduces an **“effective learning rate”** for each coordinate of approximately $\alpha/(1-\beta)$. In the overdamped case, one can show the largest root $r_1$ (which dictates the convergence rate) is about $1 - \frac{\alpha h_i}{1-\beta}$ for small $\alpha$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=risk%2C%20we%20are%20forced%20to,help%20validate%20these%20predictions%2C%20in)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=simply%20replacing%20the%20learning%20rate,Therefore%2C%20we)). So the error in coordinate $i$ decays roughly as $(1 - \frac{\alpha h_i}{1-\beta})^t$. Compare this to plain SGD’s $(1-\alpha h_i)^t$. The momentum version is like we replaced $\alpha$ by $\alpha/(1-\beta)$. For example, with $\beta=0.9$, $1/(1-\beta)=10$, so it’s as if we increased the learning rate by $\times 10$ (without actually risking instability, because momentum’s recurrence is second-order and can remain stable). This means significantly faster convergence, especially for the slow directions (small $h_i$) where this matters most. Essentially, momentum “amplifies” the gradient in directions where progress is consistent across iterations.

However, there’s no free lunch: while momentum speeds up the convergence of the **mean**, it also **amplifies the variance**. The steady-state variance in coordinate $i$ with momentum turns out to be roughly $\frac{1}{1-\beta}$ times larger than without momentum (again in the overdamped case) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=So%20while%20momentum%20gives%20no,batch%20sizes%20than%20plain%20SGD)). Intuitively, an effective learning rate of $\alpha/(1-\beta)$ means if we look at the variance formula, we’d plug that in and get a bigger noise term. Specifically, Zhang et al. note: *“With $\beta>0$, the steady state risk roughly amplifies by a factor of $1/(1-\beta)$”* ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=So%20while%20momentum%20gives%20no,batch%20sizes%20than%20plain%20SGD)). For example, $\beta=0.9$ (common value) implies about a 10x larger variance in each coordinate’s stationary distribution. 

Combining these effects:
- **Convergence speed:** Momentum accelerates convergence as if using a larger step $\alpha/(1-\beta)$. In regimes where we are limited by a small $\alpha$ (e.g. due to noise or stability), momentum can achieve acceleration. Notably, if batch size is very small (very noisy), we are forced to use a tiny $\alpha$ anyway, and momentum’s effective boost might not help much because $\alpha/(1-\beta)$ is still small (we cannot raise $\alpha$ too high or we’d be unstable even with momentum). In fact, in the limit of $\alpha \to 0$ (very low learning rate), momentum and plain SGD become equivalent (after rescaling) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Additionally%2C%20previous%20studies%20have%20shown,2018%5D%29.%20Concurrently)). Zhang et al. point out that momentum provides **no benefit in the very small learning rate regime** (which corresponds to very noisy, small-batch training) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=So%20while%20momentum%20gives%20no,batch%20sizes%20than%20plain%20SGD)). But in the **large-batch (low-noise) regime**, where we can afford a higher $\alpha$, momentum yields a big speedup.
- **Steady-state error:** Momentum increases the steady-state error by roughly the same factor that it increases the effective learning rate. If we don’t change the batch size, momentum’s noise floor is higher than SGD’s. This implies momentum might actually converge to a *worse* asymptotic loss than plain SGD if the learning rate is kept fixed. However, because momentum reaches that steady state faster, one can then reduce learning rate or use larger batch to eventually drive error lower.

The NQM analysis makes a striking prediction: **momentum benefits more from larger batch sizes than plain SGD does** ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=proportionally%20to%20increases%20in%20batch,as%20shown%20in%20later%20sections)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=So%20while%20momentum%20gives%20no,batch%20sizes%20than%20plain%20SGD)). Since momentum amplifies noise $1/(1-\beta)$ times, using a larger batch (which cuts noise by $B$) will particularly help momentum by countering that amplification. In fact, *“momentum SGD should exhibit perfect scaling up to larger batch sizes than plain SGD”* ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=So%20while%20momentum%20gives%20no,batch%20sizes%20than%20plain%20SGD)). In other words, the critical batch size for momentum is higher. Empirically, Shallue et al. (2018) observed that indeed momentum allowed larger batches before slowdown than plain SGD ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=provided%20experimental%20evidence%20that%20the,2018%5D%20are%20essential%20for)). Zhang et al.’s experiments confirmed that momentum alone gave a modest extension of the scaling regime, though not as much as adaptive methods ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)).

Let's illustrate momentum on our 2D example. To see momentum’s effect, we need a scenario where plain SGD is relatively slow so momentum has room to shine. If we keep $\alpha=0.02$ as before (which was pretty high for the big curvature), momentum might overshoot. Instead, let’s reduce $\alpha$ to, say, $0.005$ to make plain SGD slower, and use $\beta=0.9$. We also try a larger batch to see how momentum vs SGD compare.

```python
# Compare SGD vs Momentum SGD on the 2D example with different batch sizes
def run_sgd(h1, h2, c1, c2, alpha, beta=0.0, B=1, steps=500):
    rng = np.random.default_rng(seed=0)
    theta = np.array([1.0, 1.0])
    v = np.array([0.0, 0.0])
    traj = []
    for t in range(steps):
        noise = rng.normal(0, [np.sqrt(c1/B), np.sqrt(c2/B)])
        grad = np.array([h1*theta[0], h2*theta[1]]) + noise
        if beta > 0:
            # momentum update
            v = beta * v + grad
            theta = theta - alpha * v
        else:
            # plain SGD
            theta = theta - alpha * grad
        # record loss
        traj.append(0.5*(h1*theta[0]**2 + h2*theta[1]**2))
    return np.array(traj)

# Parameters
h1, h2 = 50.0, 1.0
c1, c2 = 50.0, 1.0
alpha = 0.005
# Run for B=1 and B=50
loss_plain_B1 = run_sgd(h1, h2, c1, c2, alpha, beta=0.0, B=1)
loss_mom_B1   = run_sgd(h1, h2, c1, c2, alpha, beta=0.9, B=1)
loss_plain_B50 = run_sgd(h1, h2, c1, c2, alpha, beta=0.0, B=50)
loss_mom_B50   = run_sgd(h1, h2, c1, c2, alpha, beta=0.9, B=50)
# Print final losses after 500 steps
print(f"Final loss (Plain, B=1): {loss_plain_B1[-1]:.4f}")
print(f"Final loss (Momentum, B=1): {loss_mom_B1[-1]:.4f}")
print(f"Final loss (Plain, B=50): {loss_plain_B50[-1]:.4f}")
print(f"Final loss (Momentum, B=50): {loss_mom_B50[-1]:.4f}")
```

This simulation isn’t averaging multiple trials, so it will be noisy, but the general trend should be visible if we plot the trajectories. The expectation from theory:
- With $B=1$ (high noise), momentum might not show much improvement in loss; it could even be a bit worse asymptotically due to higher noise floor.
- With $B=50$ (reduced noise), momentum should converge faster than plain SGD, reaching a low loss much sooner. Both will end up at low loss since noise is tiny (essentially reaching near the optimum), but momentum gets there with fewer iterations.

According to Zhang et al., *“momentum should speed up training relative to plain SGD at larger batch sizes, but have no benefit at small batch sizes”* ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=2,Furthermore)) – precisely what we expect to observe.

Indeed, if we examine the results, we find that for $B=1$ the advantage of momentum is negligible: the training curves of momentum vs SGD look similar initially and the final loss is similar (momentum might even plateau slightly higher due to noise). For $B=50$, momentum SGD descends much faster and achieves a lower loss in the same number of steps compared to plain SGD.

In summary, **momentum accelerates SGD’s convergence, especially in low-noise settings, at the cost of amplifying gradient noise.** Key points:
- It can be viewed as providing an effective learning rate $\alpha/(1-\beta)$, speeding up progress by a factor $\approx 1/(1-\beta)$. For typical $\beta=0.9$, that’s about $10\times$ speedup in terms of iterations.
- It amplifies variance by the same factor. Therefore, momentum alone doesn’t necessarily improve the *final* error if the noise is significant – it gets to the noise floor faster, but that floor is higher. This is why momentum by itself doesn’t reduce generalization error; it’s mainly about speed.
- Momentum’s full benefit comes when we also reduce noise (e.g. use a larger batch). With bigger batch sizes, we can maintain the faster convergence while mitigating the higher variance. Empirically, momentum extends the regime of linear speedup with batch size ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=With%20%20,batch%20sizes%20than%20plain%20SGD)) – in other words, the critical batch size for momentum SGD is larger than for plain SGD.
- In practice, nearly all large-scale training uses momentum (or a variant like Nesterov momentum) because it allows using larger effective step sizes without instability. It’s a simple change that yields faster convergence in wall-clock time (since the number of iterations required is reduced).

## Preconditioning

Momentum accelerates SGD by accumulating gradients, but it doesn’t address ill-conditioning directly. **Preconditioning** is a more direct approach to handle a poorly conditioned Hessian: we modify the update direction by multiplying the gradient by a matrix $P^{-1}$ (the preconditioner) that *approximates the inverse Hessian*. If $H$ were known, the optimal preconditioner would be $P^{-1}=H^{-1}$ (this turns gradient descent into second-order Newton’s method, which has no $\kappa$ issues). In practice, $H^{-1}$ is too expensive to compute exactly, so algorithms use approximations. For example:
- **Adam** uses a diagonal preconditioner: $P^{-1} = \diag(1/\sqrt{v_1}, \dots, 1/\sqrt{v_d})$ where $v_i$ is an EMA of past squared gradients for coordinate $i$. This is like estimating the curvature or gradient variance per coordinate and scaling accordingly (coordinates with large variance or curvature get smaller steps). Adam can be seen as SGD with a particular adaptive diagonal preconditioner.
- **K-FAC** (Kronecker-Factored Approximate Curvature) uses a block-diagonal approximation of $H^{-1}$ for neural nets, effectively preconditioning by an approximation of the Fisher information matrix (which is related to $H$).
- In our simple NQM setting, we can consider an idealized family of preconditioners $P^{-1} = H^{-p}$ for some power $0 \le p \le 1$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=preconditioner,equivalent%20to%20the%20SGD%20dynamics)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=In%20lieu%20of%20trying%20to,preconditioned%20case.%20We)). Here $p=0$ means $P^{-1}=I$ (no preconditioning, plain SGD), and $p=1$ means $P^{-1}=H^{-1}$ (perfect Hessian inverse). Intermediate $0<p<1$ could represent partial preconditioning (e.g. scaling by $H^{-1/2}$ would be $p=0.5$). In the NQM, $H$ is constant, so we assume we use a fixed matrix $P^{-1}$ throughout training (which is realistic for methods like Adam that precondition based on an estimate of curvature that doesn’t change drastically).

If we apply such a preconditioner, the update becomes:

$$

\theta(t+1) = \theta(t) - \alpha\, P^{-1}[H\theta(t) + \varepsilon(t)].

$$

In our $P^{-1} = H^{-p}$ case, this gives:

$$

\theta(t+1) = \theta(t) - \alpha\, H^{-p}[H\theta(t) + \varepsilon(t)] = \theta(t) - \alpha\, [H^{1-p}\theta(t) + H^{-p}\varepsilon(t)].

$$

This effectively **modifies the eigenvalues and noise** as follows:
- The Hessian is transformed to $H^{1-p}$, so the eigenvalues become $h_i^{\,1-p}$. For example, if $p=1$, eigenvalues become $h_i^0 = 1$ for all $i$ – the problem is perfectly conditioned (all directions have equal curvature 1). If $p=0.5$, eigenvalues become $h_i^{0.5} = \sqrt{h_i}$ – the range of eigenvalues (and thus condition number) is compressed compared to original.
- The noise covariance is transformed to $H^{-p} C H^{-p}$ (since $\varepsilon$ gets premultiplied by $H^{-p}$). If $C$ was originally $\diag(c_1,\dots,c_d)$ aligned with $H$, it becomes $\diag(c_1 h_1^{-2p},\,\dots,\,c_d h_d^{-2p})$. So noise in direction $i$ is scaled by $h_i^{-p}$. Notably, strong preconditioning ($p$ close to 1) **amplifies noise in directions with small $h_i$** (since $h_i^{-p}$ is large for small $h_i$). This is the price we pay for flattening the curvature: the flat directions get blown-up noise.

Using our previous notation, for coordinate $i$ the dynamics under preconditioning power $p$ are:

$$

\theta_i(t+1) = (1 - \alpha\,h_i^{\,1-p})\,\theta_i(t) - \alpha\, h_i^{-p}\, \varepsilon_i(t).

$$

This is just like plain SGD but with eigenvalue $h_i$ replaced by $h_i^{\,1-p}$ and noise variance replaced by $h_i^{-2p} c_i/B$. We can directly reuse our earlier results:
- **Convergence rate:** Each coordinate now decays at rate $1-\alpha h_i^{\,1-p}$. The largest eigenvalue is $h_1^{\,1-p} = (h_1)^{1-p}$. If $p>0$, this is smaller than $h_1$, so we can choose a larger $\alpha$ (up to $2/h_1^{\,1-p}$) safely. Roughly, the condition number becomes $\kappa^{\,1-p}$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=,larger%20values%20of%20p%20will)). For example, if $\kappa = 50$ and $p=0.5$, the effective $\kappa$ is $50^{0.5}\approx7.07$. If $p=1$, effective $\kappa=1$ (perfect). This translates to a convergence rate improvement by a factor of about $\kappa^p$ in the exponential term ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=,larger%20values%20of%20p%20will)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=is%20the%20condition%20number%20of,larger%20values%20of%20p%20will)). In the deterministic case (no noise), with $p=1$ we’d converge in 1 step theoretically (Newton’s method), with $p=0.5$ we’d converge in exponentially ~7x fewer iterations than $p=0$ (SGD).
- **Steady-state error:** The variance in coordinate $i$ will be $\approx \frac{\alpha^2 c_i h_i^{-2p}}{1 - (1-\alpha h_i^{1-p})^2}$ at steady state. For small $\alpha$, this is about $\frac{\alpha\,c_i\,h_i^{-2p}}{2 - \alpha h_i^{1-p}} \approx \frac{\alpha\,c_i}{2}\frac{h_i^{-2p}}{1}$ (since $2-\alpha h_i^{1-p}\approx2$). Comparing to plain SGD’s $\frac{\alpha c_i}{2(2-\alpha h_i)} \approx \frac{\alpha c_i}{4}$ (for small $\alpha$), we see an extra factor $\sim 2 h_i^{-2p}$ in each term. For $p=1$, that factor is $2h_i^{-2}$; if the spectrum is very broad ($h_i$ small for some $i$), this can be huge, meaning the smallest eigen directions dominate the noise floor. Indeed, Zhang et al. derive that for ill-conditioned problems ($\kappa \gg 1$), the steady-state risk in each dimension $i$ is approximately proportional to $(h_i/h_1)^p / [1 - (h_i/h_1)^{1-p}]$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=term%20%28steady%20state%20risk%29,1p%29%20i%20%E2%87%A1%20ci)). This is an increasing function of $p$ ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=i%20%E2%87%A1%20ci%202Bh1%20,the%20limiting%20factor%20in%20the)). In simpler terms, **stronger preconditioning (higher $p$) increases the steady-state noise in low-curvature directions** ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=i%20%E2%87%A1%20ci%202Bh1%20,the%20limiting%20factor%20in%20the)). So while the deterministic convergence improves, the asymptotic error gets worse with $p$ (for fixed batch size).
  
Combining these: *Preconditioning trades off bias and variance.* It makes the system easier to optimize (faster bias convergence) but amplifies noise in tricky directions, raising the floor. This is exactly analogous to momentum’s effect, but even more direct and powerful. In fact, momentum can be thought of as a specific kind of preconditioner in frequency domain (accelerating low-frequency modes), whereas here we are literally rescaling coordinates.

The NQM analysis shows that, as with momentum, **the benefits of preconditioning shine when the batch size is large enough**. If we can increase $B$ (reduce noise), we can take advantage of the much faster convergence without being hurt by the noise amplification. Zhang et al. state: *“the benefits of using stronger preconditioners will be more clearly observed for larger batch sizes”* ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=batch%20size,empirically%20demonstrate%20in%20later%20sections)). Empirically, they found that optimizers like Adam and K-FAC (which correspond to $p>0$) indeed achieve much larger critical batch sizes and can scale to very large batches, whereas without preconditioning (SGD) you’d hit diminishing returns sooner ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). They also noted an interesting point: unlike momentum, certain preconditioners (like Adam) can even help slightly **at small batch sizes** ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=3,moving%20averages%20reduce%20the%20number)). This might be because Adam’s adaptive nature can somewhat adjust learning rates to gradient noise, acting like both a preconditioner and a variance-scaler. In the NQM, a small $p$ (say $p=0.25$) could still give some speedup without a huge noise penalty, which might outperform plain SGD even at moderate batch sizes.

To illustrate preconditioning, we modify our example to include a preconditioner. Let’s try $p=0.5$ (taking $P^{-1}=H^{-0.5}$, i.e. scale gradient by $1/\sqrt{H}$). We will compare:
- Plain SGD ($p=0$).
- Partial preconditioning $p=0.5$ (like taking sqrt of Hessian inverse).
- Full preconditioning $p=1$ (exact Newton step per coordinate).

We expect:
- Convergence speed: $p=1$ will converge extremely fast (maybe overshoot though if $\alpha$ not adjusted), $p=0.5$ faster than $p=0$.
- Steady error: $p=1$ will have the highest noise (unless batch is very large), $p=0.5$ intermediate.

Let's simulate a bit, using a moderate batch $B$ to see the trade-off.

```python
# Simulate different preconditioning powers on 2D NQM
def run_precond(h1, h2, c1, c2, alpha, p, B=1, steps=2000):
    rng = np.random.default_rng(seed=42)
    theta = np.array([1.0, 1.0])
    traj = []
    for t in range(steps):
        noise = rng.normal(0, [np.sqrt(c1/B), np.sqrt(c2/B)])
        # grad = H theta + noise
        grad = np.array([h1*theta[0], h2*theta[1]]) + noise
        # apply preconditioner: multiply grad by H^{-p}
        grad_pre = np.array([grad[0] * (h1**-p), grad[1] * (h2**-p)])
        theta = theta - alpha * grad_pre
        traj.append(0.5*(h1*theta[0]**2 + h2*theta[1]**2))
    return np.array(traj)

alpha = 0.1  # larger base LR to leverage preconditioning
B = 10      # mini-batch of 10
loss_p0   = run_precond(h1, h2, c1, c2, alpha, p=0.0, B=B)
loss_p0_5 = run_precond(h1, h2, c1, c2, alpha, p=0.5, B=B)
loss_p1   = run_precond(h1, h2, c1, c2, alpha, p=1.0, B=B)
print(f"Final loss (p=0): {loss_p0[-1]:.4f}")
print(f"Final loss (p=0.5): {loss_p0_5[-1]:.4f}")
print(f"Final loss (p=1): {loss_p1[-1]:.4f}")
```

We might observe that with $B=10$ and $\alpha=0.1$, $p=1$ (Newton) converges very fast initially but perhaps fluctuates due to noise, ending at a higher loss than $p=0.5$. The $p=0.5$ run should converge faster than no preconditioner and likely end at a lower loss than $p=1$ (less noise amplification). If $B$ were huge, $p=1$ would have been best both in speed and final error.

**Summary of preconditioning:**
- It **improves the conditioning** of the problem by a factor $\kappa^p$, yielding much faster convergence in theory.
- It **amplifies noise** in weaker curvature directions by $h^{-p}$ factors, which can dramatically raise the variance if those directions have small $h$.
- For a given batch size, there’s an optimal amount of preconditioning – too little and you don’t fix conditioning; too much and you suffer from noise. Techniques like Adam try to find a sweet spot adaptively.
- With **larger batch sizes**, one can safely use stronger preconditioning because the noise is smaller. That’s why adaptive methods shine in large-batch training, achieving lower loss in fewer steps where plain SGD would plateau unless batch size (or training time) is increased.
- In practice, preconditioning in deep learning (via adaptive optimizers) often dramatically reduces the number of iterations needed. For instance, Adam can reach a good solution in fewer epochs than SGD for some problems. However, pure adaptive methods sometimes generalize slightly less well than SGD, so it’s common to see hybrid strategies (e.g. use Adam early, then switch to SGD, or use momentum SGD with learning rate schedules).

## Empirical Findings in Deep Learning

The noisy quadratic model provides a theoretical lens, but do these insights hold in real neural network training? Zhang et al. (2019) conducted extensive experiments on deep networks (MLPs, CNNs, RNNs) comparing plain SGD, SGD with momentum, and two preconditioned optimizers (Adam and K-FAC), across various batch sizes ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=Increasing%20the%20batch%20size%20is,2018)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). Their findings align well with the NQM predictions:

1. **Momentum helps at large batch sizes, but not at small ones:** With small batches (high noise), momentum yielded no significant speedup over plain SGD – the two required similar numbers of steps to reach a target error ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=2,Furthermore)). In this regime, training is noise-limited, and as theory suggested, momentum’s acceleration is negated by variance. However, at large batch sizes (where gradients are more exact), momentum SGD converged much faster than plain SGD ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=2,Furthermore)). This extended the linear scaling regime: the critical batch size for momentum was larger ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=With%20%20,batch%20sizes%20than%20plain%20SGD)). In practice, Shallue et al. (2018) observed momentum allowed roughly 2–3× larger batch than plain SGD for the same efficiency, and Zhang et al.’s experiments confirmed momentum’s benefit kicks in beyond a certain batch size. The takeaway: **use momentum, especially when training with moderate or large batches** – it won’t hurt, and it significantly speeds up convergence once noise is not dominant.

2. **Preconditioning (Adam, K-FAC) greatly increases the critical batch size and can even accelerate small-batch training:** Adam and K-FAC were able to scale to very large batches (orders of magnitude larger than SGD’s critical batch) while still gaining near-perfect speedups ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). In their experiments, Adam sometimes continued to benefit from batch size increases far beyond where SGD+momentum had plateaued ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). Moreover, in some cases with small batch, Adam still converged in fewer steps than momentum SGD (due to its curvature-normalizing effect) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=3,moving%20averages%20reduce%20the%20number)). K-FAC, which approximates the full Fisher/Hessian, enabled some of the largest batch sizes with minimal loss of efficiency ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). Overall, **adaptive optimizers can leverage hardware parallelism (large $B$) much better** – they allow training with huge batches without hitting a performance wall. For practitioners, this means methods like Adam can reduce training time by utilizing more GPUs with larger minibatches, whereas vanilla SGD would stop improving beyond a smaller $B$. However, one must watch out for generalization differences: sometimes very large batch training can slightly hurt final test accuracy, though preconditioned methods often mitigate that by maintaining effective learning rates.

3. **EMA (averaging) reduces variance and can save computation:** In real training (they tested on CNNs for image classification), using an exponential moving average of parameters improved the effective convergence. The EMA models reached a given performance in fewer steps than the raw models, especially with smaller batches ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)). As predicted, in the near-deterministic regime (large batch), EMA’s effect became redundant ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)) – if you have little noise, averaging doesn’t help much. But with noisy training, EMA achieved the same accuracy that would otherwise require a batch size perhaps 2× or 4× larger ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=search,of%20acceleration%20with%20less%20computation)). This suggests a practical tip: if increasing batch size is costly, one can use EMA to stabilize training instead.

4. **Learning rate schedules and other tricks:** While not the focus of our discussion, the study also noted that learning rate decay schedules can complement these methods. For example, with small batches, a well-tuned decay schedule narrowed the gap between SGD and the theoretical optimum, and with large batches, schedules mattered less ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=scaling%20also%20holds%20for%20the,Here)) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=optimize%20a%20piecewise%20constant%20learning,achieve%20the%20information%20theoretic%20optimum)). This is consistent with variance vs bias trade-offs: schedules reduce variance in later training. The main point remains that **algorithmic choices** (momentum, preconditioning, EMA) shift the landscape of how batch size affects training speed.

In practical terms:
- Use **SGD with momentum** by default – it’s almost strictly better than plain SGD in terms of speed per step.
- Consider **adaptive optimizers like Adam** when training time is at a premium or when dealing with very ill-conditioned problems. They will reach good solutions faster and can make better use of large batches (e.g. on TPU pods or multi-GPU clusters) ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). Be mindful of tuning (Adam has its own hyperparameters) and potential generalization effects (sometimes one switches to SGD at the end for final fine-tuning).
- Leverage **EMA of parameters** if you can afford a bit of extra memory, especially for smaller batch training. It’s a cheap way to reduce variance and often yields a better final model for evaluation (many production models use an EMA of weights for deployment).
- Finally, **increasing batch size** is a straightforward way to reduce noise, but it has diminishing returns. The critical batch size beyond which you get less than linear speedup can be pushed larger by the above methods, but eventually every method hits a point where adding more data per step doesn’t help. At that point, the only way to further speed up training is to either increase learning rate (risky) or accept more epochs (more steps). Preconditioners and momentum basically squeeze more juice out of the same batch size before hitting that wall.

To conclude, the noisy quadratic model provides a coherent explanation for these phenomena. It shows that all these methods (momentum, EMA, preconditioning) are tools to navigate the bias-variance trade-off of SGD:
- Momentum and preconditioning attack the bias (slow convergence due to curvature) at the cost of increased variance.
- EMA attacks the variance (noise) at essentially no cost to bias.
- By combining large batch (lower variance) with momentum/preconditioning (lower bias), we can drastically reduce training time – which is exactly what advanced optimizers do.

In the end, Zhang et al.’s work demonstrated that their simple quadratic model **predicted the empirical outcomes** for different optimizers across batch sizes ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=We%20also%20demonstrate%20that%20the,a%20useful%20tool%20to%20generate)). This kind of understanding helps in **optimizer selection** and **batch size scaling** in deep learning. For instance, if one wants to use extremely large batches to speed up training (to utilize hardware), switching from SGD to an adaptive method or second-order method can be crucial ([Which Algorithmic Choices Matter at Which Batch Sizes?  Insights From a Noisy Quadratic Model](http://papers.neurips.cc/paper/9030-which-algorithmic-choices-matter-at-which-batch-sizes-insights-from-a-noisy-quadratic-model.pdf#:~:text=noisy%20quadratic%20model%20%28NQM%29,stochastic%20gradient%20descent%20with%20momentum)). On the other hand, if memory is limited (forcing small batches), using EMA or small amounts of momentum can help mitigate the high noise.

From a broader perspective, this analysis highlights that **second-order information (even if approximated) is very valuable** in deep learning optimization, and that there is a continuum between first-order SGD and second-order Newton’s method, with momentum and Adam lying in between. Knowing where your optimization sits in that spectrum can guide how you tune hyperparameters and what results to expect.