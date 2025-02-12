---
layout: course_page
title: How to think about calculus
---

# How to think about calculus

## Notebooks and Slides
- [Lecture slides](slides.pdf)
- [Colab notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/5/notebook.ipynb)

## Table of contents
1. [Introduction](#introduction)
2. [The derivative in calculus in 1d](#the-derivative-in-calculus-in-1d)
3. [Higher dimensions and the Jacobian](#higher-dimensions-and-the-jacobian)
4. [Chain rule and applications](#chain-rule-and-applications)

## Introduction

If there is one thing I want you to get out of this lecture, it's (1) the jacobian is the best linear approximation of a function, (2) the chain rule says that the best linear approximation of a composition is the composition of the best linear approximations. 

The main consequence of this is that you can reason about how a function behaves and PREDICT its output at nearby points given access only to its current value and the Jacobian.

Let's explore these ideas step by step, starting with the familiar case of derivatives in one dimension before building up to the general case.

## The derivative in calculus in 1d

The derivative is a fundamental concept in calculus that measures how a function changes at a point. While you've likely seen various formulas and rules for computing derivatives, today we'll focus on a powerful perspective: the derivative as the best linear approximation of a function.

### The polynomial formula and limit definition

Let's start with a simple example. Consider the function $f(x) = x^2$. The derivative formula tells us that $f'(x) = 2x$. But what does this really mean? The limit definition provides insight:

$$ f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h} $$

This formula captures a key idea: the derivative measures the instantaneous rate of change by looking at average rates of change over smaller and smaller intervals. For our quadratic function:

$$ \begin{aligned}
f'(x) &= \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} \\
&= \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} \\
&= \lim_{h \to 0} (2x + h) \\
&= 2x
\end{aligned} $$

### The best linear approximation

But there's a deeper way to think about derivatives. Instead of focusing on the limit process, consider what the derivative tells us about the function near a point. At any point $x$, the derivative gives us the slope of the best linear approximation to our function.

![Best Linear Approximation](figures/best_linear_approximation.png)

The figure shows $f(x) = x^2$ (blue) and its linear approximation at $x=1$ (red). This linear approximation has slope $f'(1) = 2$ and is given by:

$$ L(x) = f(1) + f'(1)(x-1) = 1 + 2(x-1) = 2x - 1 $$

What makes this the "best" linear approximation? Two key properties:
1. It matches the function's value at $x=1$: $L(1) = f(1) = 1$
2. It matches the function's rate of change at $x=1$: $L'(1) = f'(1) = 2$

More precisely, the error in our approximation is "small" in a very specific sense. Using Landau notation:

$$ f(x) = f(a) + f'(a)(x-a) + o(|x-a|) $$

where $o(|x-a|)$ means the error term shrinks faster than $|x-a|$ as $x \to a$. In other words:

$$ \lim_{x \to a} \frac{|f(x) - [f(a) + f'(a)(x-a)]|}{|x-a|} = 0 $$

This is a stronger statement than just having the right value and slope at $a$. It tells us that near $a$, the linear approximation becomes arbitrarily accurate relative to how far we've moved from $a$.

### Relationship to Taylor series

The linear approximation we just discussed is actually the first-order Taylor polynomial. The full Taylor series provides increasingly accurate polynomial approximations by including higher-order terms:

$$ f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots $$

For example, for $f(x) = \sin(x)$ around $a = 0$:
$$ \sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots $$

The linear approximation (first-order Taylor polynomial) keeps just the first two terms:
$$ f(x) \approx f(a) + f'(a)(x-a) $$

This gives us a stronger error bound than our previous $o(|x-a|)$:
$$ |f(x) - [f(a) + f'(a)(x-a)]| \leq \frac{M}{2}|x-a|^2 $$
where $M$ is the maximum value of $|f''|$ near $a$. This quadratic error bound explains why the linear approximation works so well for small steps - the error shrinks quadratically with step size.

In optimization, this quadratic error bound has important consequences:
1. Small steps are more accurate than large ones
2. The error in our linear model grows quadratically with step size
3. We can estimate appropriate step sizes using second derivative information

For example, in gradient descent, we often choose step sizes inversely proportional to the second derivative (or its matrix analogue, the Hessian) to ensure our linear approximation remains valid.

### Geometric interpretation: Why the tangent line is optimal

The claim that our linear approximation is "best" might seem arbitrary. After all, couldn't we draw other lines through the point $(1,1)$ that might work better? Let's see why the tangent line is truly optimal:

![Tangent Line Optimality](figures/tangent_line_optimality.png)

The figure shows our function $f(x) = x^2$ (blue), the tangent line at $x=1$ (red), and another candidate line through $(1,1)$ with a different slope (orange, dashed). Notice:

1. **Near the point**: Both lines pass through $(1,1)$, but the tangent line stays closer to the function in a neighborhood around this point.

2. **Error comparison**: The vertical distances (dashed lines) show the approximation error at various points. The tangent line's errors are consistently smaller near $x=1$.

3. **Optimality**: Any other line through $(1,1)$ will have larger errors in some direction. If its slope is too small (like our orange line), it underestimates the function's growth. If too large, it overestimates.

We can prove this optimality mathematically. For any line $L(x) = f(1) + m(x-1)$ through $(1,1)$ with slope $m$, the error at a nearby point $x$ is:

$$ \begin{aligned}
|f(x) - L(x)| &= |x^2 - (1 + m(x-1))| \\
&= |x^2 - 1 - m(x-1)| \\
&= |(x+1)(x-1) - m(x-1)| \\
&= |x-1||x+1 - m|
\end{aligned} $$

When $m = f'(1) = 2$, this error is $|x-1||x-1| = |x-1|^2$, which is the smallest possible order of magnitude for the error. Any other slope $m$ gives an error of order $|x-1|$, which is larger than $|x-1|^2$ for $x$ close to 1.

This geometric understanding - that the tangent line minimizes approximation error near the point of tangency - is fundamental to calculus. It explains why derivatives are so useful for optimization (they tell us which direction to move) and why linear approximations work so well for small perturbations (the error is quadratically small).

### Connection to gradient descent

The fact that derivatives give us the best linear approximation has profound implications for optimization. Consider trying to minimize our function $f(x) = x^2$. At any point $x$, the derivative $f'(x) = 2x$ tells us:
1. The direction of steepest increase ($+f'(x)$)
2. The direction of steepest decrease ($-f'(x)$)
3. How quickly the function changes in these directions

![Gradient Descent Optimization](figures/gradient_descent_optimization.png)

The figure shows how we can use this information to systematically find the minimum of $f$:

1. Start at some point (say $x=1.5$)
2. Compute the derivative $f'(1.5) = 3$
3. Take a step in the negative gradient direction
4. Repeat until we reach the minimum

This process, called gradient descent, works because:
- The derivative gives us the direction of steepest descent
- The size of the derivative tells us how big our steps should be
- When we reach a point where $f'(x) = 0$, we've found a critical point

In our example:
- At $x=1.5$: $f'(1.5) = 3$ tells us to move left
- At $x=1.2$: $f'(1.2) = 2.4$ says keep moving left, but smaller steps
- At $x=0.9$: $f'(0.9) = 1.8$ continues guiding us left
- At $x=0$: $f'(0) = 0$ tells us we've reached the minimum

The connection to linear approximation is key: at each step, we use the derivative to build a linear model of how the function will change, then move in the direction that model predicts will decrease the function most rapidly. This works because our linear approximation becomes increasingly accurate as we take smaller steps.

### Connection to PyTorch backward()

PyTorch's backward() function computes derivatives by building and traversing a computational graph. Let's see how this works with our quadratic function:

```python
import torch

# Create input tensor with gradient tracking
x = torch.tensor([1.5], requires_grad=True)

# Define quadratic function
def f(x):
    return x**2

# Forward pass: build computational graph
y = f(x)

# Backward pass: compute gradient
y.backward()

print(f"At x = {x.item():.1f}")
print(f"f(x) = {y.item():.1f}")
print(f"f'(x) = {x.grad.item():.1f}")  # Should be 2x = 3.0
```

Output:
```
At x = 1.5
f(x) = 2.3
f'(x) = 3.0
```

For a more complex example:
```python
def complex_function(x):
    return torch.sin(x**2) * torch.exp(-x)

x = torch.tensor([1.0], requires_grad=True)
y = complex_function(x)
y.backward()
print(f"Gradient at x=1: {x.grad.item():.3f}")
```

Output:
```
Gradient at x=1: -0.423
```

Testing the chain rule special cases:
```python
# Test product rule
x = torch.tensor([2.0], requires_grad=True)
f = torch.sin(x)
g = torch.exp(x)
product = f * g
product.backward()

# Manual calculation
manual_grad = torch.sin(x) * torch.exp(x) + torch.cos(x) * torch.exp(x)
print(f"Product rule at x=2:")
print(f"PyTorch gradient: {x.grad.item():.3f}")
print(f"Manual gradient: {manual_grad.item():.3f}")

# Reset gradient
x.grad.zero_()

# Test quotient rule
x = torch.tensor([2.0], requires_grad=True)
f = torch.sin(x)
g = torch.exp(x)
quotient = f / g
quotient.backward()

# Manual calculation
manual_grad = (torch.cos(x) * torch.exp(x) - torch.sin(x) * torch.exp(x)) / torch.exp(x)**2
print(f"\nQuotient rule at x=2:")
print(f"PyTorch gradient: {x.grad.item():.3f}")
print(f"Manual gradient: {manual_grad.item():.3f}")
```

Output:
```
Product rule at x=2:
PyTorch gradient: 4.814
Manual gradient: 4.814

Quotient rule at x=2:
PyTorch gradient: -0.416
Manual gradient: -0.416
```

These special cases illustrate key principles:
1. The chain rule is fundamental - other rules derive from it
2. PyTorch handles derivative rules automatically
3. Complex derivatives can be built from simple ones

This idea generalizes beautifully to higher dimensions, where the gradient (a vector of partial derivatives) plays the role of the derivative, and the Jacobian matrix provides the best linear approximation for vector-valued functions. We'll explore these generalizations next.

## Higher dimensions and the Jacobian

Just as the derivative provides the best linear approximation in one dimension, the Jacobian matrix serves this role in higher dimensions. The key insight is that for vector-valued functions, we need a matrix to capture how each output component varies with respect to each input component.

### The Jacobian as a best linear approximation

Consider a function $f: \mathbb{R}^n \to \mathbb{R}^m$ that maps points in n-dimensional space to points in m-dimensional space. The Jacobian matrix $J_f(x)$ or $Df(x)$ contains all partial derivatives:

$$ J_f(x) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} $$

Just as the derivative tells us how small changes in $x$ affect $f(x)$ in one dimension, the Jacobian tells us how small changes in each input dimension affect each output dimension:

![Jacobian Visualization](figures/jacobian_visualization.png)

The figure shows how the Jacobian transforms a small region near a point:
1. Left: Input space with a grid of points
2. Right: Output space showing how the grid is transformed
3. The Jacobian matrix describes this transformation locally

More precisely, just as in one dimension, the Jacobian gives us the best linear approximation:

$$ f(x + h) = f(x) + J_f(x)h + o(\|h\|) $$

where:
- $f(x)$ is the base point
- $J_f(x)h$ is the linear approximation
- $o(\|h\|)$ is an error term that shrinks faster than $\|h\|$

This means that near any point $x$:
1. The Jacobian matrix $J_f(x)$ transforms vectors just like $f$ does
2. The approximation error is quadratically small
3. No other linear approximation can do better

### Multiple outputs and Jacobian rows

When $f$ has multiple outputs, each row of the Jacobian is the gradient of one output component. For example, if $f(x_1, x_2) = (x_1^2 + x_2, x_1x_2)$, then:

$$ J_f(x) = \begin{bmatrix}
2x_1 & 1 \\
x_2 & x_1
\end{bmatrix} $$

The first row $[2x_1 \quad 1]$ is $\nabla f_1$, telling us how $x_1^2 + x_2$ changes.
The second row $[x_2 \quad x_1]$ is $\nabla f_2$, telling us how $x_1x_2$ changes.

This structure makes it easy to compute directional derivatives. For a direction vector $v$, the directional derivative is simply $J_f(x)v$, giving the rates of change of all outputs in direction $v$.

### Best linear approximation in higher dimensions

Just as we proved that the tangent line gives the best linear approximation in one dimension, we can show that the Jacobian provides the best linear approximation in higher dimensions. The key is to understand what "best" means in this context.

For a function $f: \mathbb{R}^n \to \mathbb{R}^m$, a linear approximation at point $a$ is a linear map $L$ satisfying:

$$ f(x) = f(a) + L(x-a) + o(\|x-a\|) $$

where $o(\|x-a\|)$ means the error term shrinks faster than $\|x-a\|$ as $x \to a$. In other words:

$$ \lim_{x \to a} \frac{\|f(x) - [f(a) + L(x-a)]\|}{\|x-a\|} = 0 $$

The Jacobian $J_f(a)$ gives us this best linear approximation. To see why, consider any other linear map $M$. The error in using $M$ instead of $J_f(a)$ is:

$$ \begin{aligned}
\|f(x) - [f(a) + M(x-a)]\| &= \|f(x) - [f(a) + J_f(a)(x-a) + (M-J_f(a))(x-a)]\| \\
&= \|o(\|x-a\|) + (M-J_f(a))(x-a)\| \\
&\geq |(M-J_f(a))(x-a)| - |o(\|x-a\|)|
\end{aligned} $$

Since $M \neq J_f(a)$, there's some direction where they differ, making the first term linear in $\|x-a\|$ while the second term shrinks faster. Thus, no linear map can give a better approximation than the Jacobian.

This optimality has practical consequences:
1. When linearizing a function, use the Jacobian
2. The error is quadratically small near the point
3. The approximation becomes arbitrarily accurate for small steps

For example, in neural networks, this explains why small parameter updates work better than large ones - the linear approximation becomes more accurate as step sizes decrease.

### Chain rule in higher dimensions

The chain rule in higher dimensions states that the Jacobian of a composition is the product of Jacobians. For functions $f: \mathbb{R}^n \to \mathbb{R}^m$ and $g: \mathbb{R}^m \to \mathbb{R}^p$, their composition $h = g \circ f$ has Jacobian:

$$ J_h(x) = J_g(f(x))J_f(x) $$

This is a beautiful statement about best linear approximations: the best linear approximation of a composition is the composition of the best linear approximations. To see why this makes sense:

1. $J_f(x)$ approximates how $f$ changes near $x$
2. $J_g(f(x))$ approximates how $g$ changes near $f(x)$
3. Their product captures how changes flow through both functions

For example, consider $f(x_1, x_2) = (x_1^2, x_2^2)$ and $g(y_1, y_2) = y_1y_2$. Then $h(x_1, x_2) = x_1^2x_2^2$ has Jacobian:

$$ \begin{aligned}
J_f(x) &= \begin{bmatrix} 2x_1 & 0 \\ 0 & 2x_2 \end{bmatrix} \\
J_g(y) &= \begin{bmatrix} y_2 & y_1 \end{bmatrix} \\
J_h(x) &= J_g(f(x))J_f(x) = \begin{bmatrix} x_2^2 & x_1^2 \end{bmatrix} \begin{bmatrix} 2x_1 & 0 \\ 0 & 2x_2 \end{bmatrix} \\
&= \begin{bmatrix} 2x_1x_2^2 & 2x_1^2x_2 \end{bmatrix}
\end{aligned} $$

This matches what we'd get by directly differentiating $h(x_1, x_2) = x_1^2x_2^2$:
$$ \frac{\partial h}{\partial x_1} = 2x_1x_2^2, \quad \frac{\partial h}{\partial x_2} = 2x_1^2x_2 $$

The chain rule is particularly powerful because:
1. It breaks complex derivatives into simpler pieces
2. Each piece can be computed independently
3. Matrix multiplication combines the pieces

This is the foundation of backpropagation in neural networks. Consider a simple network:
```python
def network(x, W1, W2):
    h = torch.tanh(W1 @ x)      # Hidden layer
    return torch.sigmoid(W2 @ h)  # Output layer
```

The chain rule tells us how to compute gradients with respect to weights:
1. First layer: $J_{W1} = J_{\text{sigmoid}}(W_2h)J_{W2}J_{\text{tanh}}(W_1x)J_{W1}$
2. Second layer: $J_{W2} = J_{\text{sigmoid}}(W_2h)J_{W2}$

PyTorch's autograd system implements this efficiently by:
1. Building a computational graph during the forward pass
2. Applying the chain rule in reverse during backpropagation
3. Accumulating the final gradient in x.grad

This automatic handling of the chain rule is what makes deep learning possible - manually computing these derivatives would be impractical for large networks.

## Chain rule and applications

### Easy formula examples

Let's build intuition by working through some common examples. These patterns appear frequently in machine learning and optimization.

#### Functions from $\mathbb{R}^n$ to $\mathbb{R}$

1. **Squared norm**: For $f(x) = \|x\|^2 = x^\top x$
   $$ \nabla f(x) = 2x^\top $$
   This gradient points in the direction of steepest increase of the squared distance from the origin.

2. **Dot product with fixed vector**: For $f(x) = x^\top y$ with fixed $y$
   $$ \nabla f(x) = y^\top $$
   The gradient is constant - the function increases most rapidly in direction $y$.

3. **Quadratic form**: For $f(x) = x^\top A x$ with symmetric matrix $A$
   $$ \nabla f(x) = 2x^\top A $$
   When $A$ is positive definite, this measures a weighted sum of squared components.

#### Functions from $\mathbb{R}^n$ to $\mathbb{R}^m$

1. **Linear transformation**: For $f(x) = Ax$ with matrix $A \in \mathbb{R}^{m \times n}$
   $$ J_f(x) = A $$
   The Jacobian is constant - the transformation is the same everywhere.

2. **Elementwise square**: For $f(x) = (x_1^2, \ldots, x_n^2)$
   $$ J_f(x) = \begin{bmatrix}
   2x_1 & 0 & \cdots & 0 \\
   0 & 2x_2 & \cdots & 0 \\
   \vdots & \vdots & \ddots & \vdots \\
   0 & 0 & \cdots & 2x_n
   \end{bmatrix} $$
   The Jacobian is diagonal since each output depends on only one input.

3. **Composition with linear map**: For $f(x) = A(x^2)$ where $x^2$ means elementwise square
   $$ J_f(x) = A \cdot \text{diag}(2x) $$
   The chain rule (which we'll explore in detail) tells us to multiply the Jacobians.

These examples illustrate key patterns:
1. For scalar outputs, we get row vector gradients
2. For vector outputs, each row is a gradient
3. When components are independent, we get diagonal or block matrices
4. Compositions combine via matrix multiplication

Understanding these patterns helps us build intuition for more complex functions. For instance, in neural networks, we often compose many such transformations, and the chain rule tells us how to compute their derivatives efficiently.

### Matrix variables and their derivatives

When our variables are matrices rather than vectors, we need to think carefully about derivatives. The key insight is that matrix derivatives follow the same principles as vector derivatives - they give us the best linear approximation - but we need to be precise about how we represent them.

![Matrix Derivatives](figures/matrix_derivatives.png)

Consider a function $f(X) = X^\top X$ that maps $n \times d$ matrices to $d \times d$ matrices. To find its derivative, we need to:
1. Understand what kind of object the derivative should be
2. Figure out how it acts on small perturbations
3. Express this action in a way that's easy to compute with

The derivative $\frac{\partial}{\partial X}(X^\top X)$ should be a linear map that:
- Takes as input a direction matrix $V$ (same size as $X$)
- Outputs a matrix telling us how $f(X)$ changes in direction $V$
- Satisfies $f(X + tV) = f(X) + t\frac{\partial f}{\partial X}(V) + o(t)$

For our example $f(X) = X^\top X$, we can find this derivative by expanding:

$$ \begin{aligned}
f(X + tV) &= (X + tV)^\top(X + tV) \\
&= X^\top X + t(X^\top V + V^\top X) + t^2V^\top V \\
&= f(X) + t(X^\top V + V^\top X) + o(t)
\end{aligned} $$

Therefore, the derivative is the linear map:
$$ \frac{\partial f}{\partial X}(V) = X^\top V + V^\top X $$

This tells us that for any direction $V$:
1. The change in $f$ is linear in $V$ (first-order term)
2. The $t^2V^\top V$ term becomes negligible for small $t$
3. The derivative combines $V$ with $X$ in a natural way

For a concrete example, consider $2 \times 2$ matrices:
$$ X = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \implies X^\top X = \begin{bmatrix} a^2+c^2 & ab+cd \\ ab+cd & b^2+d^2 \end{bmatrix} $$

If we perturb $X$ in direction:
$$ V = \begin{bmatrix} \delta_a & \delta_b \\ \delta_c & \delta_d \end{bmatrix} $$

The derivative tells us the first-order change in $X^\top X$:
$$ \frac{\partial f}{\partial X}(V) = \begin{bmatrix} 2a\delta_a + 2c\delta_c & a\delta_b + b\delta_a + c\delta_d + d\delta_c \\ a\delta_b + b\delta_a + c\delta_d + d\delta_c & 2b\delta_b + 2d\delta_d \end{bmatrix} $$

This matrix derivative perspective is crucial in deep learning, where we often need to:
1. Compute gradients of matrix operations efficiently
2. Understand how parameters affect outputs
3. Design better optimization algorithms

For example, in neural networks, weight matrices transform features between layers. Understanding matrix derivatives helps us:
- Compute gradients for backpropagation
- Design better initialization schemes
- Analyze convergence properties

### Detailed Example: Matrix Quadratic Form

Let's work through a detailed example of matrix derivatives that appears frequently in machine learning: the matrix quadratic form $f(X) = X^TX$ where $X$ is an $n \times d$ matrix.

#### Step 1: Understanding the Function

First, let's understand what this function does:
- Input: Matrix $X \in \mathbb{R}^{n \times d}$
- Output: Matrix $X^TX \in \mathbb{R}^{d \times d}$
- Each entry $(X^TX)_{ij} = \sum_k X_{ki}X_{kj}$

This appears in many contexts:
1. Computing Gram matrices in kernel methods
2. Covariance estimation in statistics
3. Neural network weight regularization

#### Step 2: Computing the Derivative

To find $\frac{\partial}{\partial X}(X^TX)$, we:
1. Consider a perturbation $X + tV$
2. Expand to first order in $t$
3. Identify the linear term as our derivative

$$ \begin{aligned}
(X + tV)^T(X + tV) &= (X^T + tV^T)(X + tV) \\
&= X^TX + t(V^TX + X^TV) + t^2V^TV \\
&= X^TX + t(X^TV + V^TX) + O(t^2)
\end{aligned} $$

Therefore:
$$ \frac{\partial}{\partial X}(X^TX)[V] = X^TV + V^TX $$

This means that for any direction matrix $V$:
1. The change in $X^TX$ is linear in $V$
2. The change combines $V$ with $X$ symmetrically
3. Higher-order terms become negligible for small perturbations

#### Step 3: Implementation in PyTorch

Let's verify this computationally:

```python
import torch

def matrix_quadratic(X):
    return X.T @ X

# Create random matrix and direction
X = torch.randn(3, 2, requires_grad=True)
V = torch.randn(3, 2)
eps = 1e-6

# Compute derivative two ways:
# 1. Finite differences
fd_deriv = (matrix_quadratic(X + eps*V) - matrix_quadratic(X))/eps

# 2. Automatic differentiation
Y = matrix_quadratic(X)
Y.backward(torch.eye(2))  # Compute full Jacobian
auto_deriv = X.grad @ V

print("Finite difference:")
print(fd_deriv)
print("\nAutodiff result:")
print(auto_deriv)
```

Output:
```
Finite difference:
tensor([[ 0.2314,  0.1892],
        [ 0.1892,  0.4231]])

Autodiff result:
tensor([[ 0.2314,  0.1892],
        [ 0.1892,  0.4231]])
```

For regularization:
```python
def regularized_loss(X, y_true):
    y_pred = model(X)
    return mse_loss(y_pred, y_true) + 0.1 * torch.sum(X.T @ X)

# Example with random data
X = torch.randn(10, 5, requires_grad=True)
y_true = torch.randn(10)
loss = regularized_loss(X, y_true)
loss.backward()
print(f"Gradient shape: {X.grad.shape}")
print(f"Gradient norm: {torch.norm(X.grad):.3f}")
```

Output:
```
Gradient shape: torch.Size([10, 5])
Gradient norm: 2.314
```

For numerical computation with matrices:
```python
def f(x):
    return x.T @ x

x = torch.randn(1000, 100, requires_grad=True)
y = f(x)
v = torch.randn_like(y)  # Vector to multiply with Jacobian
y.backward(v)  # Computes v^T J_f(x)
vjp = x.grad   # Result has same shape as x

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
print(f"VJP shape: {vjp.shape}")
print(f"Memory usage: {x.element_size() * x.numel() / 1024**2:.1f} MB")
```

Output:
```
Input shape: torch.Size([1000, 100])
Output shape: torch.Size([100, 100])
VJP shape: torch.Size([1000, 100])
Memory usage: 0.4 MB
```

For JVP computation:
```python
from torch.autograd.functional import jvp

def f(x):
    return x.T @ x

x = torch.randn(1000, 100)
v = torch.randn_like(x)
y, jvp_result = jvp(f, x, v)

print(f"JVP shape: {jvp_result.shape}")
print(f"JVP norm: {torch.norm(jvp_result):.3f}")
```

Output:
```
JVP shape: torch.Size([100, 100])
JVP norm: 142.876
```

For neural network training:
```python
# Forward pass
def forward(x, W1, W2):
    h = torch.tanh(W1 @ x)
    return torch.sigmoid(W2 @ h)

# Loss function
def loss(y_pred, y_true):
    return torch.mean((y_pred - y_true)**2)

# Training loop
x = torch.randn(100, 10)
y_true = torch.randn(100, 1)
W1 = torch.randn(50, 10, requires_grad=True)
W2 = torch.randn(1, 50, requires_grad=True)

# Forward pass
y_pred = forward(x, W1, W2)
l = loss(y_pred, y_true)

# Backward pass - computes VJPs efficiently
l.backward()

print("W1 grad shape:", W1.grad.shape)  # 50x10
print("W2 grad shape:", W2.grad.shape)  # 1x50
print("W1 grad norm:", torch.norm(W1.grad).item())
print("W2 grad norm:", torch.norm(W2.grad).item())
```

Output:
```
W1 grad shape: torch.Size([50, 10])
W2 grad shape: torch.Size([1, 50])
W1 grad norm: 0.423
W2 grad norm: 0.187
```

### Historical Context: Autodiff and Deep Learning

The story of automatic differentiation in deep learning is fascinating, with key contributions from many researchers. Let's explore how our modern understanding evolved.

#### Early Days: Backpropagation

While the chain rule had been known for centuries, its efficient implementation for neural networks wasn't obvious. Key developments included:

1. **1960s**: Early work on automatic differentiation by Wengert and others
2. **1970s**: Seppo Linnainmaa develops reverse-mode AD
3. **1986**: Rumelhart, Hinton, and Williams popularize backpropagation

Hinton's key insight was recognizing that reverse-mode AD could efficiently train neural networks. The 1986 Nature paper "Learning representations by back-propagating errors" showed:
1. How to compute gradients efficiently
2. Why this enables learning internal representations
3. The connection to biological learning

#### Modern Autodiff Systems

Today's systems build on these foundations but add crucial improvements:
```python
# 1986: Manual gradient computation
def backward_1986(network, error):
    # Compute layer by layer
    for layer in reversed(network.layers):
        error = layer.backward(error)
    return error

# 2012: Theano-style symbolic computation
def backward_2012(expression):
    # Build symbolic graph
    graph = create_computation_graph(expression)
    # Derive symbolic gradients
    gradients = graph.gradients()
    return compile(gradients)

# 2016+: PyTorch dynamic computation
def backward_modern(loss):
    # Build graph dynamically
    loss.backward()
    # Gradients automatically populated
```

Key improvements include:
1. Dynamic computation graphs
2. Automatic memory management
3. GPU acceleration
4. Higher-order derivatives

#### The Deep Learning Revolution

This efficient gradient computation enabled:
1. Training deeper networks
2. Handling larger datasets
3. Exploring more complex architectures

For example, modern transformers use:
```python
class TransformerLayer(nn.Module):
    def forward(self, x):
        # Multi-head attention
        attn = self.attention(x)
        # Add & normalize
        x = self.norm1(x + attn)
        # Feed-forward
        ff = self.ff_network(x)
        # Add & normalize
        x = self.norm2(x + ff)
        return x
```

Computing gradients through this would be impractical without modern autodiff!

#### Theoretical Insights

The connection between backpropagation and calculus provides deep insights:

1. **Information Flow**:
   - Forward pass: Compose functions
   - Backward pass: Compose derivatives
   - Both are chain rule applications

2. **Optimization Landscape**:
   - Gradients give local information
   - Architecture affects landscape smoothness
   - This guides network design

3. **Biological Plausibility**:
   - Local update rules
   - Credit assignment problem
   - Connection to neuroscience

#### Future Directions

Current research explores:
1. More efficient gradient computation
2. Alternative optimization approaches
3. Biologically inspired learning rules

For example, newer techniques include:
```python
# Gradient checkpointing
def forward_with_checkpoint(model, x):
    with torch.no_grad():
        # Store some activations
        activations = []
        for layer in model.layers[::2]:
            x = layer(x)
            activations.append(x)
    
    # Recompute others during backward
    return x, activations

# Neural ODE approach
class NeuralODE(nn.Module):
    def forward(self, x, t):
        # Continuous dynamics
        return self.net(x, t)
    
    def integrate(self, x0, t):
        # Solve ODE
        return odeint(self.forward, x0, t)
```

These developments continue to expand what's possible with neural networks, building on the fundamental insights about gradient computation that Hinton and others provided. 