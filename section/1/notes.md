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

## 5. Implementation in pure Python: derivatives by hand

We will implement gradient descent using ordinary Python floats. In 1D, the loop is small enough that you can see every moving piece.

We will start with the quadratic objective

$$
\begin{aligned}
f(x) &= \tfrac{1}{2}x^2 \\
f'(x) &= x
\end{aligned}
$$

### A minimal gradient descent loop

A minimal loop has the following ingredients:

1. **Initialize** a starting point $x_0$.
2. Repeat:
   - compute the loss $f(x)$,
   - compute the derivative $f'(x)$,
   - update $x \leftarrow x - \eta f'(x)$,
   - log what happened (so you can debug and tune),
   - stop when a rule triggers.

### Minimal version: update + printing

```python
# Save as: script/gd_1d_python_minimal.py

def f(x: float) -> float:
    return 0.5 * x * x

def df(x: float) -> float:
    return x

def main():
    x = 5.0
    eta = 0.5
    max_iters = 10

    for k in range(max_iters):
        fx = f(x)
        gx = df(x)
        print(f"k={k:2d}  x={x:+.6f}  f(x)={fx:.3e}  |f'(x)|={abs(gx):.3e}")
        x = x - eta * gx

    print(f"final x={x:+.6f}  final f(x)={f(x):.3e}")

if __name__ == "__main__":
    main()
```

### Full version: logging + stopping + a diagnostics plot

```python
# Save as: script/gd_1d_python.py

import os
import matplotlib.pyplot as plt


def gradient_descent_1d(f, df, x0, eta, max_iters=200, eps_grad=1e-8, eps_obj=None):
    """
    1D gradient descent with simple logging.

    Stops when:
      - k reaches max_iters, or
      - |f'(x)| <= eps_grad, or
      - f(x) <= eps_obj (if eps_obj is not None)

    Returns:
      x_final (float), hist (dict of lists)
    """
    x = float(x0)

    hist = {"k": [], "x": [], "f": [], "abs_df": []}

    for k in range(max_iters):
        fx = float(f(x))
        gx = float(df(x))

        hist["k"].append(k)
        hist["x"].append(x)
        hist["f"].append(fx)
        hist["abs_df"].append(abs(gx))

        if eps_grad is not None and abs(gx) <= eps_grad:
            break
        if eps_obj is not None and fx <= eps_obj:
            break

        x = x - eta * gx

    return x, hist


def save_diagnostics_plot(hist, outpath, title):
    k = hist["k"]
    fvals = hist["f"]
    gabs = hist["abs_df"]

    plt.figure(figsize=(6.5, 3.5))
    plt.semilogy(k, fvals, label="objective f(x_k)")
    plt.semilogy(k, gabs, label="|f'(x_k)|")
    plt.xlabel("iteration k")
    plt.ylabel("value (semilog y)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    # Example: f(x) = 1/2 x^2, f'(x) = x
    def f(x): return 0.5 * x * x
    def df(x): return x

    x0 = 5.0
    eta = 0.5

    x_final, hist = gradient_descent_1d(f, df, x0=x0, eta=eta, max_iters=80, eps_grad=1e-10)
    print(f"Final x: {x_final:.6e}")
    print(f"Final f(x): {hist['f'][-1]:.6e}")
    print(f"Iterations: {len(hist['k'])}")

    save_diagnostics_plot(
        hist,
        outpath="figures/gd_python_quadratic_diagnostics.png",
        title="GD on f(x)=1/2 x^2 (pure Python)",
    )


if __name__ == "__main__":
    main()
```

For $f(x)=\tfrac{1}{2}x^2$, the objective and the derivative are directly related:

$$
f(x)=\tfrac{1}{2}x^2=\tfrac{1}{2}\,|f'(x)|^2
$$

So stopping by small objective and stopping by small derivative are closely aligned in this specific example.

![Gradient descent diagnostics for the quadratic](figures/gd_python_quadratic_diagnostics.png)
*Figure 1.3: On the quadratic, both the objective and the derivative magnitude decay geometrically when the step size is stable.*

### Changing the loss means changing the derivative

A hand-written loop hard-codes two ingredients:

1. how to compute the loss $f(x)$,
2. how to compute the derivative $f'(x)$.

If you change the loss, you must update both.

Two examples:

1. Shifted quadratic:

$$
\begin{aligned}
f(x) &= \tfrac{1}{2}(x-1)^2 \\
f'(x) &= x-1
\end{aligned}
$$

2. Double well:

$$
\begin{aligned}
f(x) &= \tfrac{1}{2}(x^2-1)^2 \\
f'(x) &= 2x(x^2-1)
\end{aligned}
$$

For simple formulas this is manageable. As soon as the loss becomes a long composition of operations, it becomes easy to make a derivative mistake. That is where automatic differentiation becomes valuable.

## 6. PyTorch basics: tensors, `requires_grad`, and `backward()`

PyTorch can compute derivatives automatically. You write the loss as code, and PyTorch produces the derivative with respect to variables you mark as trackable.

In this lecture we will treat a **single scalar** $x$ as our parameter. In PyTorch, even a scalar is represented as a tensor.

### What is a tensor here?

A tensor is a container for numbers. In this section:

- `x` will be a scalar tensor (think: “a number with extra bookkeeping”),
- we will ask PyTorch to store $\frac{d}{dx} f(x)$ in `x.grad`.

### A single derivative via `backward()`

```python
import torch

x = torch.tensor(2.0, requires_grad=True)  # track derivatives w.r.t. x
loss = 0.5 * x**2                          # f(x) = 1/2 x^2

loss.backward()                             # compute d(loss)/dx

print("x =", x.item())
print("loss =", loss.item())
print("d(loss)/dx =", x.grad.item())        # should be 2.0
```

What to remember:

- `requires_grad=True` tells PyTorch to compute derivatives with respect to `x`.
- `loss.backward()` computes the derivative and stores it in `x.grad`.
- `x.grad` is a tensor; `.item()` turns it into a Python float for printing.

### Sanity check (recommended before writing any loop)

Compare to an analytic derivative at one point.

For the double well,

$$
\begin{aligned}
f(x) &= \tfrac{1}{2}(x^2-1)^2 \\
f'(x) &= 2x(x^2-1)
\end{aligned}
$$

```python
import torch

def loss_fn(x):
    return 0.5 * (x**2 - 1.0) ** 2

x = torch.tensor(2.0, requires_grad=True)
loss = loss_fn(x)
loss.backward()

autograd_val = x.grad.item()
analytic_val = 2.0 * 2.0 * (2.0**2 - 1.0)   # 2x(x^2-1) at x=2

print("autograd:", autograd_val)
print("analytic:", analytic_val)
```

This is a quick consistency check before you write a full loop.

## 7. Autodiff under the hood, common pitfalls, and the training loop

### Recorded operations + chain rule (what `backward()` is doing)

When you compute a loss from `x`, PyTorch records the sequence of operations used to build that loss. During `backward()`, it applies the chain rule in reverse order.

Example loss:

$$
f(x)=\tfrac{1}{2}(x^2-1)^2
$$

One way to view this as a composition:

- $u(x)=x^2$
- $v(u)=u-1$
- $w(v)=\tfrac{1}{2}v^2$

Then $f = w \circ v \circ u$, so

$$
f'(x)=w'(v(u(x)))\,v'(u(x))\,u'(x)
$$

PyTorch performs this same multiplication of local derivatives, but it does it automatically from the recorded operations.

A sketch of the forward computation:

```
x  ->  u = x^2  ->  v = u - 1  ->  loss = 0.5 * v^2
```

### Pitfall A: gradients accumulate unless you clear them

If you call `backward()` multiple times, PyTorch adds into `x.grad`. For “one gradient per step” loops, you must clear `x.grad` each iteration.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

(0.5 * x**2).backward()
print("after first backward, x.grad =", x.grad.item())   # 2.0

(0.5 * (x - 1.0)**2).backward()
print("after second backward, x.grad =", x.grad.item())  # accumulated

x.grad = None
(0.5 * (x - 1.0)**2).backward()
print("after clearing, x.grad =", x.grad.item())         # correct for the last loss
```

### Pitfall B: tracking the update step breaks the loop

A tempting update is:

```python
x = x - eta * x.grad
```

This creates a *new* tensor `x` built from tracked operations. In a beginner loop, the next iteration typically fails because the gradient you expect to read from `x.grad` is no longer populated.

A tiny failing example:

```python
import torch

eta = 0.1
x = torch.tensor(2.0, requires_grad=True)

# Step 1
x.grad = None
loss = 0.5 * x**2
loss.backward()
print("step 1 grad:", x.grad.item())

# Wrong update: records the update in the graph and replaces x
x = x - eta * x.grad

# Step 2
x.grad = None
loss = 0.5 * x**2
loss.backward()

print("step 2 grad:", x.grad)  # typically None in this beginner pattern
```

Correct fix: update without tracking.

```python
with torch.no_grad():
    x -= eta * x.grad
```

### Pitfall C: calling `backward()` twice on the same recorded operations

If you call `backward()` twice on the same loss object, PyTorch raises an error.

In gradient descent loops, the solution is simple: **recompute the loss each iteration** (new forward pass), call `backward()` once, update, repeat.

### A complete 1D gradient descent loop in PyTorch

Now we can write a clean loop that:

- clears `x.grad`,
- computes the loss,
- calls `backward()`,
- updates under `torch.no_grad()`.

```python
# Save as: script/gd_1d_torch.py

import os
import torch
import matplotlib.pyplot as plt


def gd_1d_torch(loss_fn, x0, eta, max_iters=200, eps_grad=1e-8, eps_obj=None):
    x = torch.tensor(float(x0), requires_grad=True)

    hist = {"k": [], "x": [], "loss": [], "abs_grad": []}

    for k in range(max_iters):
        x.grad = None                 # clear accumulation

        loss = loss_fn(x)             # forward pass
        loss.backward()               # backward pass

        g = x.grad.item()
        l = loss.item()

        hist["k"].append(k)
        hist["x"].append(x.item())
        hist["loss"].append(l)
        hist["abs_grad"].append(abs(g))

        if eps_grad is not None and abs(g) <= eps_grad:
            break
        if eps_obj is not None and l <= eps_obj:
            break

        with torch.no_grad():
            x -= eta * x.grad

    return x.item(), hist


def save_diagnostics_plot(hist, outpath, title):
    k = hist["k"]
    loss_vals = hist["loss"]
    gabs = hist["abs_grad"]

    plt.figure(figsize=(6.5, 3.5))
    plt.semilogy(k, loss_vals, label="loss f(x_k)")
    plt.semilogy(k, gabs, label="|df/dx at x_k|")
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
    eta_dw = 0.02

    # Loss 1: quadratic
    def loss1(x): return 0.5 * x**2
    x_final, hist = gd_1d_torch(loss1, x0=x0, eta=eta, max_iters=80, eps_grad=1e-10)
    print(f"[quadratic]  final x={x_final:.6e}, final loss={hist['loss'][-1]:.6e}, iters={len(hist['k'])}")
    save_diagnostics_plot(hist, "figures/gd_torch_quadratic_diagnostics.png", "GD on f(x)=1/2 x^2 (PyTorch)")

    # Loss 2: shifted quadratic (no derivative code changes)
    def loss2(x): return 0.5 * (x - 1.0) ** 2
    x_final, hist = gd_1d_torch(loss2, x0=x0, eta=eta, max_iters=80, eps_grad=1e-10)
    print(f"[shifted]    final x={x_final:.6e}, final loss={hist['loss'][-1]:.6e}, iters={len(hist['k'])}")

    # Loss 3: double well (no derivative code changes)
    def loss3(x): return 0.5 * (x**2 - 1.0) ** 2
    x_final, hist = gd_1d_torch(loss3, x0=x0, eta=eta_dw, max_iters=200, eps_grad=1e-10)
    print(f"[doublewell] final x={x_final:.6e}, final loss={hist['loss'][-1]:.6e}, iters={len(hist['k'])}")


if __name__ == "__main__":
    main()
```

Only the definition of `loss_fn` changes across these examples. For the double well, we reduce the step size for stability. The loop structure stays the same.

![PyTorch diagnostics for the quadratic](figures/gd_torch_quadratic_diagnostics.png)
*Figure 1.4: Same diagnostics as Figure 1.3, but derivatives are produced by autodiff.*

## 8. Tuning: choosing a step size

The step size $\eta$ controls the tradeoff between:

- moving aggressively (fewer iterations when stable),
- not overshooting (avoiding divergence).

### Quadratic example: exact recursion

Take

$$
\begin{aligned}
f(x) &= \tfrac{1}{2}x^2 \\
f'(x) &= x
\end{aligned}
$$

Gradient descent gives:

$$
x_{k+1}=x_k-\eta x_k = (1-\eta)x_k
$$

So:

- If $\|1-\eta\|<1$, then $x_k \to 0$ geometrically.
- If $\|1-\eta\|>1$, then $\|x_k\|$ grows geometrically (divergence).

For this specific quadratic, the stability condition is:

$$
0 < \eta < 2
$$

A concrete divergence example: if $\eta=3$, then $x_{k+1}=-2x_k$, so $\|x_k\|$ doubles every step.

### Compare small, reasonable, and too-large step sizes

We will compare:

- $\eta=0.001$ (stable but slow),
- $\eta=0.5$ (stable and fast),
- $\eta=3$ (diverges).

![Effect of step size on GD for the quadratic](figures/gd_stepsize_comparison_quadratic.png)
*Figure 1.5: Same objective, same initialization, different step sizes. Small $\eta$ makes slow progress. Large $\eta$ diverges.*

### A simple tuning protocol: time-to-result

Fix a target:

$$
f(x_k) \le 10^{-5}
$$

Define “time-to-result” as the number of iterations needed to hit this threshold (with a max-iteration cap).

For the quadratic, the recursion shows $\eta=1$ is special: it sends $x_1=0$ from any $x_0$.

You can still do the empirical version: sweep a range of step sizes and record the iteration count.

![Step size sweep on the quadratic](figures/stepsize_sweep_quadratic.png)
*Figure 1.6: A brute-force sweep of step sizes. For this quadratic, the winner is near $\eta=1$.*

### Repeat on a nonconvex objective

For the double well,

$$
f(x)=\tfrac{1}{2}(x^2-1)^2
$$
there are two global minimizers ($x=-1$ and $x=1$), and the curvature depends on where you are. A fixed step size can behave differently depending on initialization.

We can still run the same tuning sweep, but we must interpret it carefully: we are measuring “time-to-reach a minimizer” from a fixed starting point.

![Step size sweep on the double well](figures/stepsize_sweep_doublewell.png)
*Figure 1.7: The best fixed step size depends on the objective and the initialization. For nonconvex problems, sweeping hyperparameters is often the pragmatic baseline.*

## 9. Conclusion

What you should take from this lecture:

- An optimization problem is: decision variable + objective + constraints.
- Minimizers and stationary points are different targets. In nonconvex settings, stationarity is the default guarantee.
- Iterative algorithms should be judged by diagnostics and by time-to-result, not by whether they run.
- Gradient descent is a local method derived from the first-order Taylor model.
- With hand-written derivatives, changing the loss forces you to update derivative code too.
- With autodiff, you can change the loss without rewriting derivatives, as long as you follow a few PyTorch conventions:
  - mark variables with `requires_grad=True`,
  - clear `x.grad` each iteration,
  - update parameters under `torch.no_grad()`.

Next lecture: we move from full gradients to **stochastic gradient descent**, where gradients are estimated from samples.
