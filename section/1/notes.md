---
layout: course_page
title: Optimization Terminology, Philosophy, and Basics in 1D
---

# 1. Optimization Terminology, Philosophy, and Basics in 1D

[cheatsheet](cheatsheet.html)

## Notebooks and Slides
- [Lecture slides](slides.pdf)
- [Colab notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/1/notebook.ipynb)
- [Live demo notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/1/live-demo.ipynb)

## Table of contents
1. [Setup: what is an optimization problem?](#1-setup-what-is-an-optimization-problem)
2. [Solutions, optimal values, and stationarity](#2-solutions-optimal-values-and-stationarity)
3. [Iterative algorithms and diagnostics](#3-iterative-algorithms-and-diagnostics)
4. [Gradient descent in 1D via the local model](#4-gradient-descent-in-1d-via-the-local-model)
5. [Implementation in NumPy: gradients by hand](#5-implementation-in-numpy-gradients-by-hand)
6. [Implementation in PyTorch: gradients by `backward()`](#6-implementation-in-pytorch-gradients-by-backward)
7. [Autograd under the hood: recorded operations + chain rule](#7-autograd-under-the-hood-recorded-operations--chain-rule)
8. [Tuning: choosing a step size](#8-tuning-choosing-a-step-size)
9. [Conclusion](#9-conclusion)

## 1. Setup: what is an optimization problem?

This course is about one recurring template:

1. **Decision variables:** what we are allowed to choose.
2. **Objective (loss):** what we want to make small.
3. **Algorithm:** how we update the decision variables.
4. **Diagnostics:** what we plot or log to see if the algorithm is behaving.

In this lecture we do everything in one dimension so that the objects are concrete.

### Optimization problems

An optimization problem is:

$$
\min_{x \in C} f(x)
$$

- **Decision variable:** $x \in \mathbb{R}$ (a number we choose).
- **Objective / loss:** $f:\mathbb{R}\to\mathbb{R}$ (a scalar function we can evaluate).
- **Constraint set (feasible set):** $C \subseteq \mathbb{R}$ (the allowed values of $x$).

Unconstrained problems are the special case $C=\mathbb{R}$:

$$
\min_{x \in \mathbb{R}} f(x)
$$

A simple constrained example is a bound constraint:

$$
\min_{x \ge 0} (x-1)^2
$$

Here the feasible set is $C=[0,\infty)$.

### Convex vs nonconvex losses (in 1D)

Two toy losses we will use repeatedly:

- Convex quadratic:

$$
f_{\text{quad}}(x) = \tfrac{1}{2}(x-1)^2
$$

- A nonconvex "double well":

$$
f_{\text{dw}}(x) = \tfrac{1}{2}(x^2-1)^2
$$

The scaling $\tfrac{1}{2}$ is cosmetic. It keeps some derivatives simple.

![Convex vs nonconvex 1D losses](figures/convex_nonconvex_1d.png)
*Figure 1.1: A convex quadratic (one minimizer) versus a nonconvex double well (two minimizers). Convexity is global geometry. Nonconvexity is the default in modern ML.*

<!--
CODEX PLOT TASK (Figure 1.1): Convex vs nonconvex 1D losses

Goal: generate figures/convex_nonconvex_1d.png.

1) Create a Python script at: script/plot_convex_nonconvex_1d.py
2) Use only: numpy, matplotlib (no seaborn).
3) Make sure the script:
   - creates the directories "script" and "figures" if they do not exist
   - is runnable from the lecture directory via: python script/plot_convex_nonconvex_1d.py

Plot specification:
- Use a dense grid: x = np.linspace(-2.5, 2.5, 2000).
- Define:
    f1 = 0.5*(x - 1)**2
    f2 = 0.5*(x**2 - 1)**2
- Create a single figure with 1 row and 2 columns:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
- Left panel:
    - plot f1 vs x with a clean line
    - mark the minimizer x=1 with a dot
    - title: "Convex quadratic"
- Right panel:
    - plot f2 vs x
    - mark minimizers x=-1 and x=1 with dots
    - mark the stationary point x=0 with a different marker (e.g., x or triangle)
    - title: "Nonconvex double well"
- Axes:
    - x-label on both: "x"
    - y-label on left only: "f(x)"
    - light grid on both panels
    - y-limits chosen so both plots are readable (e.g., 0 to 3)
- Styling:
    - no legend needed if titles are clear
    - use tight_layout or constrained_layout
- Save:
    figures/convex_nonconvex_1d.png
    dpi=200
    bbox_inches="tight"
- Do not show the plot in an interactive window (no plt.show()).
-->

## 2. Solutions, optimal values, and stationarity

### Minimizers and optimal value

A point $x^\ast$ is a **(global) minimizer** if $x^\ast \in C$ and

$$
f(x^\ast) \le f(x)
\quad \text{for all } x \in C
$$

The **optimal value** is

$$
f^\ast = \inf_{x \in C} f(x)
$$

If a minimizer $x^\ast$ exists, then $f^\ast = f(x^\ast)$.

In many problems we care about **approximate minimizers**: points $x$ whose objective value is close to $f^\ast$. A standard definition is:

$$
f(x) - f^\ast \le \varepsilon
$$

This quantity $f(x)-f^\ast$ is the **objective gap** (also called suboptimality gap). In real ML problems we typically do not know $f^\ast$, so the objective gap is not directly observable. In our toy 1D examples we often have $f^\ast=0$, so the objective gap is just $f(x)$.

### Stationary points

For unconstrained problems with differentiable $f$, a **stationary point** is a point where the derivative is zero:

$$
f'(x)=0
$$

An **$\varepsilon$-stationary point** is a point where

$$
\|f'(x)\| \le \varepsilon
$$

One might think "minimizer means derivative is zero." However, that statement is only guaranteed for **interior** local minimizers in unconstrained problems (or more generally when the minimizer is not stuck on the boundary of the feasible set). With constraints, minimizers can sit at the boundary and have nonzero derivative. We will return to constraints later (projection methods and KKT conditions).

Nonconvex problems can have many stationary points: local minima, local maxima, and flat points. In nonconvex ML, convergence guarantees often target stationarity rather than global optimality.

## 3. Iterative algorithms and diagnostics

### Iterative algorithms

An iterative method produces a sequence

$$
x_0, x_1, x_2, \dots
$$

Each $x_k$ is an **approximate solution candidate**. The algorithm usually has a small number of knobs (hyperparameters). In this lecture the main knob is the **step size** (learning rate) $\eta$.

### What we monitor

Two common diagnostics are:

1. **Objective values:** $f(x_k)$ (or objective gap $f(x_k)-f^\ast$ when $f^\ast$ is known).
2. **Gradient norm:** in 1D this is $\|f'(x_k)\|$.

Since we're trying to minimize $f$, we'd like it to decrease along the iterate sequence. If we also know $f^\ast$ we can also tell how well we're solving the problem. However in general $f^\ast$ is not known, so we can never quite tell how well we're solving the problem.

Gradient norms are a slightly weaker metric. For certain problems, we can show that whenever $\|f'(x_k)\|$ is small we also have that $f(x_k) - f^\ast$ is small. However, characterizing the precise relationship between the two is difficult and, in general, small gradients do not mean the objective is small. On the other hand, gradients are at least always computable and always tend to zero as our iterates approach minimizers, so they're still a useful diagnostic.

### Termination criteria

Common stopping rules:

- Stop after a fixed number of iterations $T$.
- Stop when $\|f'(x_k)\| \le \varepsilon_{\text{grad}}$.
- Stop when $f(x_k)-f^\ast \le \varepsilon_{\text{obj}}$ (only when $f^\ast$ is known).
- Stop when progress stalls (plateauing diagnostics).

In our toy problem we will combine a max-iteration cap with a threshold on either the objective or the gradient.

### What makes an algorithm "good"?

Two basic resources:

- **Memory.** What state must be stored to take the next step?
  - Gradient descent stores the current iterate (and maybe a few running statistics).
  - Other methods may store a history (momentum, quasi-Newton curvature, etc.).

- **Computation.**
  - **Work per iteration:** what does one step cost?
  - **Number of iterations:** how many steps do we need to reach a target accuracy?

A useful way to talk about iteration counts is: to reach an accuracy $\varepsilon$, the algorithm needs $T(\varepsilon)$ steps. Typical scalings look like:

- **Sublinear** rates: $T(\varepsilon)$ behaves like a power of $1/\varepsilon$.
- **Linear** rates (optimization jargon): $T(\varepsilon)$ behaves like $\log(1/\varepsilon)$.

One might think "linear convergence means the error decays like a line." However, in optimization, **linear convergence means geometric decay**.

A standard model for the error is:

- Sublinear decay:

$$
e_k \approx \frac{C}{(k+1)^p}
$$

- Linear (geometric) decay:

$$
e_k \approx C \rho^k
\quad \text{for some } \rho \in (0,1)
$$

On a semilog-$y$ plot, geometric decay is a straight line. Sublinear decay bends.

![Sublinear vs linear convergence](figures/convergence_rates_semilogy.png)
*Figure 1.2: Semilog-$y$ plot of two toy error sequences. Geometric decay (called "linear convergence" in optimization) appears as a straight line. Sublinear decay bends.*

<!--
CODEX PLOT TASK (Figure 1.2): Sublinear vs linear convergence on a semilogy plot

Goal: generate figures/convergence_rates_semilogy.png.

1) Create a Python script at: script/plot_convergence_rates_semilogy.py
2) Use only: numpy, matplotlib.
3) k-range: k = np.arange(0, 201)

Define two sequences:
- sublinear: e_sub = 1.0 / (k + 1.0)
- linear (geometric): e_lin = 0.9**k

Plot:
- Use plt.semilogy(k, e_sub, label="sublinear: 1/(k+1)")
- Use plt.semilogy(k, e_lin, label="linear (geometric): 0.9^k")
- Axis labels: x-axis "iteration k", y-axis "error e_k"
- Title: "Sublinear vs linear (geometric) convergence"
- Add a legend.
- Add a light grid that works well with semilogy (major and minor if possible).
- Save to figures/convergence_rates_semilogy.png, dpi=200, bbox_inches="tight"
- No plt.show().
-->

## 4. Gradient descent in 1D via the local model

### The local model (Taylor approximation)

Let $f:\mathbb{R}\to\mathbb{R}$ be differentiable. Near a point $x$, the first-order Taylor approximation is:

$$
f(x+\Delta) \approx f(x) + f'(x)\Delta
$$

This approximation says: locally, $f$ looks like an affine function with slope $f'(x)$.

If $f'(x)>0$, then decreasing $x$ (negative $\Delta$) decreases the local model. If $f'(x)<0$, then increasing $x$ decreases the local model.

This motivates the gradient descent update.

### Gradient descent update rule (1D)

Gradient descent with step size $\eta>0$ uses:

$$
x_{k+1} = x_k - \eta f'(x_k)
$$

A basic check is to plug the update into the first-order model:

$$
f(x_k - \eta f'(x_k)) \approx f(x_k) - \eta \|f'(x_k)\|^2
$$

So for small enough $\eta$, the objective decreases to first order.

In higher dimensions, $f'(x)$ becomes $\nabla f(x)$, and $\|f'(x)\|^2$ becomes $\|\nabla f(x)\|^2$. The mechanism is the same.

## 5. Implementation in NumPy: gradients by hand

We will implement gradient descent in NumPy first. The point is not that NumPy is better. The point is that in plain NumPy you must compute derivatives yourself, so you feel what autodiff is saving you from.

### The anatomy of a minimal training loop

A minimal 1D “training loop” has the same basic parts you will see in ML code:

1. **Initialize** the parameter (here: a scalar $x_0$).
2. Repeat for $k=0,1,2,\dots$:
   - **Forward pass:** compute objective value $f(x_k)$.
   - **Gradient:** compute derivative $f'(x_k)$.
   - **Log diagnostics:** store or print $f(x_k)$ and $\|f'(x_k)\|$.
   - **Stop** if a termination rule fires.
   - **Update:** $x_{k+1} = x_k - \eta f'(x_k)$.

We will build this in three runnable versions. In class, you can overwrite the same file and rerun.

### Version 1: update rule + printing (no history, no stopping)

```python
# Save as: script/gd_1d_numpy.py  (Version 1)

import numpy as np


def f(x):
    return 0.5 * x**2


def grad_f(x):
    return x


def main():
    x = 5.0
    eta = 0.5
    max_iters = 10

    for k in range(max_iters):
        fx = f(x)
        gx = grad_f(x)
        print(f"k={k:2d}  x={x:+.6f}  f(x)={fx:.3e}  ||f'(x)||={abs(gx):.3e}")
        x = x - eta * gx

    print(f"final x={x:+.6f}  final f(x)={f(x):.3e}")


if __name__ == "__main__":
    main()
````

At this point we have the update rule and a way to see whether the iterates are moving in the right direction.

### Version 2: add logging + stopping rules

Now we add two things that show up in essentially every optimization loop:

* **a history object** (so we can plot later)
* **termination criteria**

```python
# Save as: script/gd_1d_numpy.py  (Version 2)

import numpy as np


def gradient_descent_1d(f, grad_f, x0, eta, max_iters=200, eps_grad=1e-8, eps_obj=None):
    """
    1D gradient descent with simple logging.

    Stops when:
      - k reaches max_iters, or
      - ||f'(x)|| <= eps_grad, or
      - f(x) <= eps_obj  (if eps_obj is not None)

    Returns:
      x (final iterate), hist dict
    """
    x = float(x0)

    hist = {"k": [], "x": [], "f": [], "abs_grad": []}

    for k in range(max_iters):
        fx = float(f(x))
        gx = float(grad_f(x))

        hist["k"].append(k)
        hist["x"].append(x)
        hist["f"].append(fx)
        hist["abs_grad"].append(abs(gx))

        if eps_grad is not None and abs(gx) <= eps_grad:
            break
        if eps_obj is not None and fx <= eps_obj:
            break

        x = x - eta * gx

    return x, hist


def main():
    # Example: f(x) = 1/2 x^2, f'(x) = x
    f = lambda x: 0.5 * x**2
    grad_f = lambda x: x

    x0 = 5.0
    eta = 0.5

    x_final, hist = gradient_descent_1d(f, grad_f, x0=x0, eta=eta, max_iters=80, eps_grad=1e-10)
    print(f"Final x: {x_final:.6e}")
    print(f"Final f(x): {hist['f'][-1]:.6e}")
    print(f"Iterations: {len(hist['k'])}")


if __name__ == "__main__":
    main()
```

### Version 3: add a diagnostics plot

Now we use the saved history to produce the standard “two curves” diagnostic plot.

```python
# Save as: script/gd_1d_numpy.py  (Version 3)

import os
import numpy as np
import matplotlib.pyplot as plt


def gradient_descent_1d(f, grad_f, x0, eta, max_iters=200, eps_grad=1e-8, eps_obj=None):
    x = float(x0)

    hist = {"k": [], "x": [], "f": [], "abs_grad": []}

    for k in range(max_iters):
        fx = float(f(x))
        gx = float(grad_f(x))

        hist["k"].append(k)
        hist["x"].append(x)
        hist["f"].append(fx)
        hist["abs_grad"].append(abs(gx))

        if eps_grad is not None and abs(gx) <= eps_grad:
            break
        if eps_obj is not None and fx <= eps_obj:
            break

        x = x - eta * gx

    return x, hist


def save_diagnostics_plot(hist, outpath, title):
    k = np.array(hist["k"])
    fvals = np.array(hist["f"])
    gabs = np.array(hist["abs_grad"])

    plt.figure(figsize=(6.5, 3.5))
    plt.semilogy(k, fvals, label="objective f(x_k)")
    plt.semilogy(k, gabs, label="||f'(x_k)||")
    plt.xlabel("iteration k")
    plt.ylabel("value (semilog y)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    f = lambda x: 0.5 * x**2
    grad_f = lambda x: x

    x0 = 5.0
    eta = 0.5

    x_final, hist = gradient_descent_1d(f, grad_f, x0=x0, eta=eta, max_iters=80, eps_grad=1e-10)
    print(f"Final x: {x_final:.6e}, final f(x): {hist['f'][-1]:.6e}, iterations: {len(hist['k'])}")

    save_diagnostics_plot(
        hist,
        outpath="figures/gd_numpy_quadratic_diagnostics.png",
        title="GD on f(x)=1/2 x^2 (NumPy)",
    )


if __name__ == "__main__":
    main()
```

Two remarks:

1. For $f(x)=\tfrac{1}{2}x^2$, the derivative is $f'(x)=x$.
2. In this quadratic case, objective and gradient are directly related:

$$
f(x)=\tfrac{1}{2}x^2=\tfrac{1}{2}|f'(x)|^2
$$

So stopping by objective or stopping by gradient are roughly equivalent (up to a square).

![Gradient descent diagnostics for the quadratic](figures/gd_numpy_quadratic_diagnostics.png)
*Figure 1.3: On the quadratic, both the objective and the gradient magnitude decay geometrically when the step size is stable.*

<!--
CODEX PLOT TASK (Figure 1.3): Diagnostics for GD on the quadratic (NumPy)

If you prefer a dedicated plotting script rather than using script/gd_1d_numpy.py directly, create:

- script/plot_gd_numpy_quadratic_diagnostics.py

Requirements:
- run GD on f(x)=0.5*x^2 with grad=x
- use x0=5, eta=0.5, max_iters=80, eps_grad=1e-10
- log f(x_k) and ||grad||
- semilogy plot both curves on the same axes
- title: "GD on f(x)=1/2 x^2 (NumPy)"
- save: figures/gd_numpy_quadratic_diagnostics.png (dpi=200, bbox_inches="tight")
-->

### Changing the loss means recomputing the gradient

Now suppose that $x^2$ is not the function we wish to optimize any more. To update the loop above we must change two things: the computation of the loss and the derivative.

1. Shifted quadratic:

$$
f(x)=\tfrac{1}{2}(x-1)^2
\quad \Longrightarrow \quad
f'(x)=x-1
$$

2. Double well:

$$
f(x)=\tfrac{1}{2}(x^2-1)^2
\quad \Longrightarrow \quad
f'(x)=2x(x^2-1)
$$

In NumPy, you must write these derivatives correctly each time you change $f$. 

This might not seem like a big hurdle. But for extremely complicated loss functions, it's easy to get derivatives wrong! This is one of the many reasons we'll use PyTorch in this case.

## 6. Implementation in PyTorch: gradients by `backward()`

PyTorch implements *autodifferentiation.* Tis means that we can change the loss fucntion without recomputing the derivative by hand. The price is that you must follow the rules of the autodifferentiation (sometimes called 'autograd') api.

### Version 1: a single derivative via `backward()`

This is the smallest example that shows what PyTorch is doing.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
loss = 0.5 * x**2
loss.backward()

print("x =", x.item())
print("dloss/dx =", x.grad.item())  # should be 2.0
```

A few facts that become important once we put this in a loop:

* A tensor with `requires_grad=True` is treated as a variable you want derivatives with respect to.
* After `loss.backward()`, PyTorch stores the derivative in `x.grad`.
* By default, PyTorch **accumulates** gradients into `x.grad`, so you must clear it each iteration.

### Version 2: a complete 1D training loop in PyTorch

Now we build the same loop structure as in NumPy:

* forward pass: compute `loss`
* backward pass: populate `x.grad`
* log diagnostics
* update under `no_grad()`

```python
# Save as: script/gd_1d_torch.py  (Version 2)

import os
import torch
import matplotlib.pyplot as plt


def gd_1d_torch(loss_fn, x0, eta, max_iters=200, eps_grad=1e-8, eps_obj=None):
    x = torch.tensor(float(x0), requires_grad=True)

    hist = {"k": [], "x": [], "loss": [], "abs_grad": []}

    for k in range(max_iters):
        # Clear gradient buffer (PyTorch accumulates by default)
        x.grad = None

        loss = loss_fn(x)
        loss.backward()

        g = float(x.grad.detach().cpu().item())
        l = float(loss.detach().cpu().item())

        hist["k"].append(k)
        hist["x"].append(float(x.detach().cpu().item()))
        hist["loss"].append(l)
        hist["abs_grad"].append(abs(g))

        if eps_grad is not None and abs(g) <= eps_grad:
            break
        if eps_obj is not None and l <= eps_obj:
            break

        # Update without tracking the update in autograd
        with torch.no_grad():
            x -= eta * x.grad

    return float(x.detach().cpu().item()), hist


def save_diagnostics_plot(hist, outpath, title):
    k = hist["k"]
    loss_vals = hist["loss"]
    gabs = hist["abs_grad"]

    plt.figure(figsize=(6.5, 3.5))
    plt.semilogy(k, loss_vals, label="loss f(x_k)")
    plt.semilogy(k, gabs, label="||df/dx|| at x_k")
    plt.xlabel("iteration k")
    plt.ylabel("value (semilog y)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    x0 = 5.0
    eta = 0.5

    # Loss 1: quadratic
    loss_fn = lambda x: 0.5 * x**2
    x_final, hist = gd_1d_torch(loss_fn, x0=x0, eta=eta, max_iters=80, eps_grad=1e-10)
    print(f"[quadratic] final x={x_final:.6e}, final loss={hist['loss'][-1]:.6e}, iters={len(hist['k'])}")
    save_diagnostics_plot(hist, "figures/gd_torch_quadratic_diagnostics.png", "GD on f(x)=1/2 x^2 (PyTorch)")

    # Loss 2: shifted quadratic (no gradient code changes)
    loss_fn = lambda x: 0.5 * (x - 1.0) ** 2
    x_final, hist = gd_1d_torch(loss_fn, x0=x0, eta=eta, max_iters=80, eps_grad=1e-10)
    print(f"[shifted]   final x={x_final:.6e}, final loss={hist['loss'][-1]:.6e}, iters={len(hist['k'])}")

    # Loss 3: double well (no gradient code changes)
    loss_fn = lambda x: 0.5 * (x**2 - 1.0) ** 2
    x_final, hist = gd_1d_torch(loss_fn, x0=x0, eta=eta, max_iters=200, eps_grad=1e-10)
    print(f"[doublewell] final x={x_final:.6e}, final loss={hist['loss'][-1]:.6e}, iters={len(hist['k'])}")


if __name__ == "__main__":
    main()
```

The only thing that changed across the three problems is the line defining `loss_fn`. The update rule and the derivative computation stayed the same.

![PyTorch diagnostics for the quadratic](figures/gd_torch_quadratic_diagnostics.png)
*Figure 1.4: Same diagnostics as Figure 1.3, but gradients are produced by autograd.*

### Gradient accumulation (the most common first bug)

PyTorch adds gradients into `x.grad`. That is useful when you intentionally accumulate gradients across mini-batches, but it is wrong for a vanilla "one gradient per step" loop.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

loss1 = 0.5 * x**2
loss1.backward()
print("after loss1.backward(), x.grad =", x.grad.item())  # 2.0

loss2 = 0.5 * (x - 1.0) ** 2
loss2.backward()
print("after loss2.backward(), x.grad =", x.grad.item())  # 2.0 + 1.0 = 3.0 (accumulated)

# Correct way:
x.grad = None
loss3 = 0.5 * (x - 1.0) ** 2
loss3.backward()
print("after clearing grad, x.grad =", x.grad.item())  # 1.0
```

In normal PyTorch training code, you usually clear gradients with either:

* `x.grad = None` for individual tensors, or
* `optimizer.zero_grad(set_to_none=True)` for models.

## 7. Autograd under the hood: recorded operations + chain rule

PyTorch autograd is the chain rule applied to a recorded sequence of operations.

### A 1D chain rule example

Consider:

$$
f(x)=\tfrac{1}{2}(x^2-1)^2
$$

Write it as a composition:

* $u(x)=x^2$
* $v(u)=u-1$
* $w(v)=\tfrac{1}{2}v^2$

Then $f = w \circ v \circ u$. The chain rule gives:

$$
f'(x)=w'(v(u(x)))v'(u(x))u'(x)
$$

Compute each derivative:

* $u'(x)=2x$
* $v'(u)=1$
* $w'(v)=v$

So

$$
f'(x)=(x^2-1)2x = 2x(x^2-1)
$$

Autograd does the same multiplication of local derivatives, but it does it by traversing the recorded computation graph.

### A computational graph (ASCII)

For the double well loss, the forward pass looks like:

```
x  --->  u = x^2  --->  v = u - 1  --->  loss = 0.5 * v^2
```

The backward pass walks this graph in reverse and applies the chain rule.

### Three common failure modes (and what they mean)

#### Failure mode 1: forgetting to clear gradients

Symptom: gradients are too large and the method diverges or behaves strangely.

Cause: `x.grad` stores the sum of gradients from all previous `backward()` calls unless you reset it.

Fix: clear gradients once per iteration (`x.grad = None` or `optimizer.zero_grad(set_to_none=True)`).

#### Failure mode 2: tracking the parameter update in the graph

One might think the update line

```python
x = x - eta * x.grad
```

is harmless. However, this creates a new tensor `x` whose value depends on the old `x` through differentiable operations. Two things go wrong:

1. The "iterate" becomes part of a growing graph (memory and compute blow up).
2. The new `x` is typically not a leaf tensor, so `x.grad` may stop being populated.

The standard fix is to update without tracking:

```python
with torch.no_grad():
    x -= eta * x.grad
```

If you need to deliberately break a graph, `x = x.detach()` is the explicit tool. In this lecture, `no_grad()` is the right choice.

#### Failure mode 3: calling `backward()` twice on the same graph

Symptom: you see a runtime error like "Trying to backward through the graph a second time..."

Cause: after `backward()` PyTorch frees the intermediate buffers it needed for the reverse pass. This is the default because it saves memory.

Fix (rare in standard training loops): pass `retain_graph=True` to the first backward call.

In typical optimization loops, you do not want `retain_graph=True`. You rebuild the loss each iteration, call backward once, update, repeat.

## 8. Tuning: choosing a step size

The step size $\eta$ controls the tradeoff between:

* moving aggressively (fewer iterations when stable)
* not overshooting (avoiding divergence)

### Quadratic example: exact recursion

Take:

$$
f(x)=\tfrac{1}{2}x^2
\quad \Longrightarrow \quad
f'(x)=x
$$

Gradient descent gives:

$$
x_{k+1}=x_k-\eta x_k = (1-\eta)x_k
$$

So:

* If $|1-\eta|<1$, then $x_k \to 0$ geometrically.
* If $|1-\eta|>1$, then $|x_k|$ grows geometrically (divergence).

For this specific quadratic, the stability condition is:

$$
0 < \eta < 2
$$

A concrete divergence example: if $\eta=3$, then $x_{k+1}=-2x_k$, so $|x_k|$ doubles every step.

### Compare small, reasonable, and too-large step sizes

We will compare:

* $\eta=0.001$ (stable but slow)
* $\eta=0.5$ (stable and fast)
* $\eta=3$ (diverges)

![Effect of step size on GD for the quadratic](figures/gd_stepsize_comparison_quadratic.png)
*Figure 1.5: Same objective, same initialization, different step sizes. Small $\eta$ makes slow progress. Large $\eta$ diverges.*

<!--
CODEX PLOT TASK (Figure 1.5): Step size comparison on the quadratic

Goal: generate figures/gd_stepsize_comparison_quadratic.png.

1) Create: script/plot_gd_stepsize_comparison_quadratic.py
2) Use numpy + matplotlib.
3) Setup:
   - f(x)=0.5*x**2
   - grad(x)=x
   - x0=5.0
   - max_iters=60
   - etas = [0.001, 0.5, 3.0]
4) For each eta:
   - run GD and log objective values f(x_k)
   - if objective exceeds a large threshold (e.g., 1e6) treat as diverged and stop early
5) Plot:
   - semilogy of objective vs iteration for each eta on the same axes
   - xlabel: "iteration k"
   - ylabel: "objective f(x_k) (semilog y)"
   - title: "GD on f(x)=1/2 x^2: effect of step size"
   - legend includes eta values
   - light grid
6) Save to figures/gd_stepsize_comparison_quadratic.png, dpi=200, bbox_inches="tight"
-->

### A simple tuning protocol: time-to-result

Fix a target:

$$
f(x_k) \le 10^{-5}
$$

Define "time-to-result" as the number of iterations needed to hit this threshold (with a max-iteration cap).

For the quadratic, the best step size is easy to see from the recursion: $\eta=1$ gives $x_1=0$ in one step from any $x_0$.

We can still do the empirical version: sweep a range of step sizes and record the iteration count.

![Step size sweep on the quadratic](figures/stepsize_sweep_quadratic.png)
*Figure 1.6: A brute-force sweep of step sizes. For this quadratic, the winner is near $\eta=1$ because the recursion makes the solution exact in one step.*

<!--
CODEX PLOT TASK (Figure 1.6): Step size sweep on the quadratic

Goal: generate figures/stepsize_sweep_quadratic.png.

1) Create: script/plot_stepsize_sweep_quadratic.py
2) Use numpy + matplotlib.
3) Setup:
   - f(x)=0.5*x**2
   - grad(x)=x
   - x0=5.0
   - eps_obj = 1e-5
   - max_iters = 100
   - etas = np.arange(0.001, 3.000 + 0.0001, 0.005)
4) For each eta:
   - run GD up to max_iters
   - stop early if f(x)<=eps_obj
   - if |x| grows beyond a large bound (e.g., 1e6), mark as diverged and stop
   - record:
       - iters_to_hit: the first k where f(x_k)<=eps_obj
       - if never hits, set iters_to_hit = np.nan (or max_iters+1)
5) Plot:
   - x-axis: eta
   - y-axis: iters_to_hit
   - prefer a scatter plot or thin line
   - y-limits should make the curve readable (ignore NaNs)
   - title: "Iterations to reach f(x)<=1e-5 vs step size (quadratic)"
   - axis labels: "step size eta", "iterations to threshold"
   - light grid
6) Save to figures/stepsize_sweep_quadratic.png, dpi=200, bbox_inches="tight"
-->

### Repeat on a nonconvex objective

For the double well,

$$
f(x)=\tfrac{1}{2}(x^2-1)^2
$$

there are two global minimizers ($x=-1$ and $x=1$), and the local curvature depends on where you are. A single fixed step size can behave very differently depending on initialization.

We can still run the same tuning sweep, but we must interpret it carefully: we are measuring "time-to-reach some minimizer" from a fixed starting point, not a global property of the algorithm.

![Step size sweep on the double well](figures/stepsize_sweep_doublewell.png)
*Figure 1.7: The best fixed step size depends on the objective and the initialization. For nonconvex problems, sweeping hyperparameters is often the pragmatic baseline.*

<!--
CODEX PLOT TASK (Figure 1.7): Step size sweep on the double well

Goal: generate figures/stepsize_sweep_doublewell.png.

1) Create: script/plot_stepsize_sweep_doublewell.py
2) Use numpy + matplotlib.
3) Setup:
   - f(x)=0.5*(x**2 - 1)**2
   - grad(x)=2*x*(x**2 - 1)
   - choose a fixed initialization, e.g. x0=2.0 (positive so it should converge to +1 when stable)
   - eps_obj = 1e-5
   - max_iters = 200 (use 200 here; 100 is sometimes too small for small eta)
   - etas = np.arange(0.001, 3.000 + 0.0001, 0.005)
4) For each eta:
   - run GD up to max_iters
   - stop early if f(x)<=eps_obj
   - divergence detection: if |x|>1e6 or f(x)>1e12, mark as diverged and stop
   - record iters_to_hit as before (NaN if never hits)
5) Plot:
   - scatter or line: eta vs iters_to_hit
   - title: "Iterations to reach f(x)<=1e-5 vs step size (double well)"
   - labels: "step size eta", "iterations to threshold"
   - light grid
6) Save: figures/stepsize_sweep_doublewell.png, dpi=200, bbox_inches="tight"
-->

## 9. Conclusion

What you should take from this lecture:

* An optimization problem is: decision variable + objective + constraints.
* Minimizers and stationary points are different targets. In nonconvex settings, stationarity is the default guarantee.
* Iterative algorithms should be judged by diagnostics and by time-to-result, not by whether they run.
* Gradient descent is a local method derived from the first-order Taylor model.
* In NumPy, changing the loss means changing the derivative code.
* In PyTorch, changing the loss does not require rewriting derivatives, but you must obey autograd rules (clear grads, update under `no_grad()`).

Next lecture: we move from full gradients to **stochastic gradient descent**, where gradients are estimated from samples or mini-batches.

