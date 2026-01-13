---
layout: course_page
title: Introduction (Cheatsheet)
---

## Slide 1: Lecture 0 — Introduction

**Purpose:** Set the goal for today and the course.

- Course: numerical optimization for data science and machine learning.
- Today: how a machine learning task becomes an optimization problem, and how we implement the optimizer in PyTorch.
- Notebook: [Colab notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/0/notebook.ipynb)

---

## Slide 2: Course syllabus and key points

**Purpose:** Orient you to what this course is actually about.

- We will learn to **formulate** optimization problems from machine learning problems.
- We will learn to **implement** and **debug** gradient-based methods in PyTorch.
- We will use the same spine repeatedly: decision variables, objective, update rule, diagnostics.

---

## Slide 3: Prerequisites, format, deliverables

**Purpose:** Make expectations concrete.

**Prerequisites**
- Calculus + linear algebra (Math 2400).
- Probability (Stat 4300).
- Python programming.

**Format**
- Lecture-based, with frequent group meetings with the professor.
- Weekly programming exercises in notebooks.

**Deliverables (final project)**
- Start drafting by Week 3; refine throughout the semester.
- Several checkpoints (drafts, presentations) for feedback.
- Final submission: GitHub repo (code, report, slides) plus a polished demo (for example, a Colab notebook).

---

## Slide 4: A brief history of optimization

**Purpose:** Place “optimization for ML” in a larger arc.

```

# EVOLUTION OF OPTIMIZATION

1950s                1960s-1990s              2000s                  TODAY
├─────────────────┐  ├────────────────┐  ┌────────────────┐  ┌─────────────────┐
│ LINEAR PROGRAM. │  │ CONVEX OPTIM.  │  │ SOLVER ERA     │  │ DEEP LEARNING   │
│ Dantzig's       │──│ Interior-point │--│ CVX & friends  │──│ PyTorch         │
│ Simplex Method  │  │ Large-scale    │  │ "Write it,     │  │ Custom losses   │
└─────────────────┘  └────────────────┘  │  solve it"     │  │ LLM boom        │
│                    │            └────────────────┘  └─────────────────┘
│                    │                   │                    │
▼                    ▼                   ▼                    ▼
APPLICATIONS:        APPLICATIONS:       APPLICATIONS:        APPLICATIONS:
• Logistics         • Control           • Signal Process    • Language Models
• Planning          • Networks          • Finance           • Image Gen
• Military          • Engineering       • Robotics          • RL & Control

````

- Mid-20th Century: linear programming and operations research (simplex; logistics and planning).
- 1960s–1990s: convex optimization matured (gradient methods, interior-point methods, large-scale solvers).
- 2000s: solver-era tools made standard convex problems easy to specify and solve.
- Modern era: deep learning shifted emphasis to “build and iterate” with nonconvex models, direct gradient-based methods, and custom losses.

---

## Slide 5: Why PyTorch?

**Purpose:** Explain why this course is centered on PyTorch.

- Deep learning’s success was driven in part by modern auto-differentiation frameworks.
- These frameworks let you change a model or loss quickly while keeping the same optimization loop.
- Older solver-based workflows excel for classical, well-structured problems; PyTorch excels when we want to build nonconvex models and iterate fast.

---

## Slide 6: Preview — spam classification becomes optimization

**Purpose:** Show the conversion from an ML task to an optimization problem.

We start with a task: classify email as spam or not spam.

We turn that into an optimization problem by specifying:
1. **Features:** how an email becomes a vector $x$.
2. **Decision variables:** weights $w$ that map features to a prediction.
3. **Objective:** a loss function $L(w)$ (cross-entropy).
4. **Solve:** choose $w$ by minimizing $L(w)$ using gradient descent.

---

## Slide 7: Features — email $\mapsto$ vector $x$

**Purpose:** Make “features” concrete.

Example feature set (here $x \in \mathbb{R}^5$):
- `exclamation_count`
- `urgent_words`
- `suspicious_links`
- `time_sent`
- `length`

The point is not that these are perfect features. The point is that we can turn text into numbers and then optimize over a model built on those numbers.

---

## Slide 8: Prediction rule (decision variable = $w$)

**Purpose:** Show how $w$ produces a prediction.

We score an email by a weighted sum, then convert it to a probability:

$$
p_w(x) = \sigma(x^\top w).
$$

![Spam Classification Process](figures/spam_classification_process.png)

---

## Slide 9: Sigmoid = score $\mapsto$ probability

**Purpose:** Explain the probability map we use for binary classification.

![Sigmoid Function](figures/sigmoid.png)

- Output is in $(0,1)$, so we can interpret it as a probability.
- Large positive scores give probabilities near $1$; large negative scores give probabilities near $0$.
- Score $0$ maps to probability $0.5$ (the decision boundary).

---

## Slide 10: Objective (cross-entropy) and what we minimize

**Purpose:** State the loss and make the minimization goal explicit.

Training data: $(x_i,y_i)$ with $y_i \in \{0,1\}$.

We choose $w$ by minimizing the average cross-entropy loss:

$$
L(w)=\frac{1}{n}\sum_{i=1}^n \left[-y_i\log(\sigma(x_i^\top w))-(1-y_i)\log(1-\sigma(x_i^\top w))\right].
$$

![Cross-Entropy Loss](figures/cross_entropy.png)

- The loss is small when predicted probabilities match the labels.
- Confident wrong predictions are heavily penalized, which is exactly what you want when training a classifier.

---

## Slide 11: Gradients and gradient descent

**Purpose:** Explain why the negative gradient direction is the basic update.

The gradient collects partial derivatives:

$$
\nabla L(w)=\left(\frac{\partial L}{\partial w_1},\ldots,\frac{\partial L}{\partial w_d}\right).
$$

The first-order approximation is

$$
L(w+\Delta)\approx L(w)+\langle \nabla L(w),\Delta\rangle.
$$

Taking $\Delta=-\eta \nabla L(w)$ gives the decrease-to-first-order calculation

$$
L(w-\eta \nabla L(w)) \approx L(w) - \eta \|\nabla L(w)\|^2.
$$

So gradient descent uses

$$
w \leftarrow w - \eta \nabla L(w),
\qquad
w_j \leftarrow w_j - \eta \frac{\partial L}{\partial w_j}.
$$

Here $\eta>0$ is the learning rate (stepsize).

---

## Slide 12: A picture (useful, but limited)

**Purpose:** Give intuition without replacing the mechanism.

![Gradient descent visualization showing path from high point to minimum](figures/gradient_descent.png)

This visualization is a simplification. In higher dimensions, the optimization landscape can have local minima, saddle points, and narrow valleys.

---

## Slide 13: Implementing the update in PyTorch

**Purpose:** Show that the code is implementing the math update.

In PyTorch, `loss.backward()` computes $\nabla L(w)$ and stores it in `weights.grad`. The update line is the same as $w \leftarrow w - \eta \nabla L(w)$.

```python
weights = torch.randn(5, requires_grad=True)
learning_rate = 0.01

for _ in range(1000):
    predictions = spam_score(features, weights)
    loss = cross_entropy_loss(predictions, true_labels)

    loss.backward()

    with torch.no_grad():
        weights -= learning_rate * weights.grad
        weights.grad.zero_()
````

Two details:

* `loss.backward()` fills `weights.grad` with partial derivatives.
* We call `weights.grad.zero_()` because PyTorch accumulates gradients by default.

---

## Slide 14: Numerical results (diagnostics vs generalization)

**Purpose:** Explain what the plots can and cannot tell you.

![Loss curves](figures/training_run.png)

* Loss and training accuracy are **diagnostics**: they check that the optimization loop is reducing the objective on the training set.
* Test accuracy checks whether performance **generalizes** to data not used for training.
* A large train–test gap suggests overfitting. In this run, the curves are close and stabilize, so there is no obvious generalization gap for this dataset.

---

## Slide 15: What, how, and why of PyTorch (autodiff)

**Purpose:** Explain the mechanism: recorded composition + chain rule.

If you build a scalar loss using PyTorch operations, PyTorch records the operations used to compute it. When you call `backward()`, it applies the chain rule through that recorded computation and produces derivatives with respect to variables that have `requires_grad=True`.

One-dimensional example. Fix $y$ and define

$$
f(x)=(x^2-y)^2.
$$

Write $h(x)=x^2$ and $g(z)=(z-y)^2$, so $f=g\circ h$. The chain rule gives

$$
f'(x)=g'(h(x))h'(x)=2(x^2-y)\cdot 2x = 4x(x^2-y).
$$

```python
y = 3.0
x = torch.tensor(2.0, requires_grad=True)

f = (x**2 - y)**2
f.backward()
print(x.grad.item())  # 4*x*(x**2 - y)
```

The payoff: you can change the model or the loss and keep the same training loop.

---

## Slide 16: Tentative course structure + learning outcomes

**Purpose:** Give you the roadmap and what you should be able to do by the end.

**Course structure (high-level)**

* Start in one dimension: decision variables, loss functions, gradients, GD and SGD, implementations (NumPy, then PyTorch).
* Move to higher dimensions: the linear algebra interface in PyTorch (tensors, norms, matrix products) and efficiency basics.
* A menu of problems: classic ML, deep learning, inverse problems, and a brief RL introduction.
* A menu of algorithms: GD, SGD, Adam/AdamW, and other methods that show up in current practice.
* Benchmarking and tuning: how to compare methods without fooling yourself.
* Systems and theory: GPUs/distributed training basics, and what theory can and cannot explain.

**By the end of the course, you should be able to**

1. Formulate optimization problems (variables, objectives, constraints) in math and code.
2. Implement and debug gradient-based training loops in PyTorch.
3. Choose reasonable algorithms and hyperparameters, and recognize bad tuning.
4. Benchmark methods in a way that is not misleading.
5. Have basic systems awareness (compute, memory, data loading bottlenecks).
6. Produce a portfolio-quality project (clean repo, working implementation, short write-up).

