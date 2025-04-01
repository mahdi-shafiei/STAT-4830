---
layout: course_page
title: Adaptive Optimization Methods - Cheat Sheet
---
# Adaptive Optimization Methods - Cheat Sheet

## 1. Introduction & Motivation

**Core Problem**: SGD struggles with:
- **Feature scaling sensitivity**: Oscillates across steep dimensions, slow along flat ones
- **Poor conditioning**: Condition number $\kappa = \frac{\text{largest eigenvalue}}{\text{smallest eigenvalue}}$ of Hessian
- **Difficult learning rate trade-off**: Must be small enough for steep directions, but slows progress in flat ones

![ZigZag](../3/figures/zigzag_visualization.png)

**Adaptive Methods Insight**: Give each parameter its own learning rate that adapts during training:
- Parameters with consistently large gradients → smaller learning rates
- Parameters with small gradients → larger learning rates

**Historical Development**:
- **Adagrad** (Duchi et al., 2011): Accumulates squared gradients, slows learning for frequently updated parameters
- **RMSProp** (Tieleman & Hinton, 2012): Uses exponential moving average of squared gradients
- **Adam** (Kingma & Ba, 2015): Combines RMSProp with momentum + bias correction
- **AdamW** (Loshchilov & Hutter, 2019): Decouples weight decay for better regularization

## 2. Algorithm Definitions

### Adagrad
**Update Rule**:
$$w_{t,i} = w_{t-1,i} - \frac{\alpha}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}$$

Where:
- $G_{t,ii} = \sum_{j=1}^{t} g_{j,i}^2$ (sum of squared gradients)
- $\alpha$ is global learning rate
- $\epsilon$ is small constant (~10⁻⁸) for numerical stability

**Vector Form**:
$$w_t = w_{t-1} - \alpha \cdot G_t^{-1/2} \odot g_t$$

**Key Strength**: Eliminates manual tuning for different parameters, good for sparse gradients

**Key Weakness**: $G_{t,ii}$ monotonically increases → learning rate continually shrinks → learning eventually stops

### Adam
**Update Rules**:
1. Compute moment estimates:

   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$ 

   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

2. Apply bias correction:

   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

   $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

3. Update parameters:

   $$w_t = w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Hyperparameters**:
- $\alpha$: Step size (learning rate)
- $\beta_1$: Decay rate for first moment (typically 0.9)
- $\beta_2$: Decay rate for second moment (typically 0.999)
- $\epsilon$: Small constant (~10⁻⁸)

**Key Insights**:
- $\frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}$ functions as parameter-specific adaptive learning rate
- $\hat{m}_t$ provides momentum-like smoothing
- Bias correction prevents small learning rates at beginning

### AdamW
**Key Insight**: L2 regularization ≠ weight decay for adaptive methods

**Standard Adam with L2 Regularization**:

$$w_t = w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}(\hat{m}_t + \lambda w_{t-1})$$

**AdamW Decoupled Approach**:

$$w_t = (1 - \alpha\lambda)w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$$

**Complete Update**:
1. Compute gradient: $g_t = \nabla \ell(w_{t-1})$
2. Update momentum: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3. Update velocity: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4. Bias correction: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$ and $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
5. Apply weight decay: $w_t = (1 - \alpha\lambda)w_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t$

**Key Benefits**:
- Ensures consistent regularization across all parameters
- Makes weight decay hyperparameter $\lambda$ easier to tune

### figure 
![section2_figures](figures/section2_figures.png)

## 3. Theoretical Analysis

### Common Assumptions
1. **Lower Bounded Objective**: $L(w) \geq L_* \quad \forall w \in \mathbb{R}^d$
2. **Bounded Gradients ($\ell_\infty$ norm)**: $\|\nabla \ell(w, z)\|_\infty \leq R$
3. **Smoothness**: $\|\nabla L(w) - \nabla L(w')\|_2 \leq L\|w - w'\|_2$

### Convergence Measure
- **Target**: Find stationary points where gradient is (approximately) zero
- **Measure**: Expected squared norm of gradient $\mathbb{E}[\|\nabla L(w)\|^2]$
- **Random Index**: $\tau_N$ with values in $\{0, 1, ..., N-1\}$
- **Bound Form**: $\mathbb{E}[\|\nabla L(w_{\tau_N})\|^2] \leq \frac{C(L(w_0) - L_*)}{\sqrt{N}} + \frac{D \ln(N)}{\sqrt{N}}$

### Adagrad Results
For constant step size $\alpha > 0$, $\beta_2 = 1$:

$$\mathbb{E}[\|\nabla L(w_{\tau_N})\|^2] \leq \frac{2R(L(w_0) - L_*)}{\alpha\sqrt{N}} + \frac{1}{\sqrt{N}}\left(4dR^2 + \alpha dRL\right)\ln\left(1 + \frac{NR^2}{\epsilon}\right)$$

**Key Insights**:
- Initialization term: $\frac{2R(L(w_0) - L_*)}{\alpha\sqrt{N}}$ (decreases at $O(1/\sqrt{N})$)
- Steady-state error: $\frac{1}{\sqrt{N}}\left(4dR^2 + \alpha dRL\right)\ln\left(1 + \frac{NR^2}{\epsilon}\right)$ (decreases at rate $O(\ln(N)/\sqrt{N})$)
- Dependence on dimension $d$ (unusual compared to SGD)
- No need for learning rate decay to achieve convergence

### Adam Results

For step size $\alpha_n = \alpha \sqrt{\frac{1-\beta_2^n}{1-\beta_2}}$, $0 < \beta_2 < 1$, $\beta_1 = 0$:
$$\mathbb{E}[\|\nabla L(w_{\tau_N})\|^2] \leq \frac{2R(L(w_0) - L_*)}{\alpha N} + E\left(\frac{1}{N}\ln\left(1 + \frac{R^2}{(1-\beta_2)\epsilon}\right) - \ln(\beta_2)\right)$$

Where $E = \frac{4dR^2}{\sqrt{1-\beta_2}} + \frac{\alpha dRL}{1-\beta_2}$

**Key Insights**:
- Initialization term decreases faster at $O(1/N)$ vs Adagrad's $O(1/\sqrt{N})$
- Steady-state error decreases at $O(\ln(N)/N)$
- Term $-\ln(\beta_2)$ creates "noise floor" below which Adam cannot go
- Trade-off: As $\beta_2 \to 1$, coefficient $E$ increases

### Impact of Momentum
For Adam with $0 \leq \beta_1 < \beta_2$:

$$\mathbb{E}[\|\nabla L(w_{\tau_N})\|^2] \leq \frac{2R(L(w_0) - L_*)}{\alpha\tilde{N}} + E\left(\frac{1}{\tilde{N}}\ln\left(1 + \frac{R^2}{(1-\beta_2)\epsilon}\right) - \frac{N}{\tilde{N}}\ln(\beta_2)\right)$$

Where:
- $\tilde{N} = N - \frac{\beta_1}{1-\beta_1}$
- $E = \frac{\alpha dRL(1-\beta_1)}{(1-\beta_1/\beta_2)(1-\beta_2)} + \frac{12dR^2\sqrt{1-\beta_1}}{(1-\beta_1/\beta_2)^{3/2}\sqrt{1-\beta_2}} + \frac{2\alpha^2dL^2\beta_1}{(1-\beta_1/\beta_2)(1-\beta_2)^{3/2}}$

**Key Insight**: Momentum theoretically worsens convergence guarantees (dependency $O((1-\beta_1)^{-1})$) despite empirical benefits

## 4. Practical Implementation

### PyTorch Implementation Essentials

**Adagrad**:
```python
# Core logic:
state[i].add_(grad * grad)  # Accumulate squared gradients
std = torch.sqrt(state[i] + eps)  # Add eps for stability
param.addcdiv_(grad, std, value=-lr)  # Update with scaled gradient
```

**Adam**:
```python
# Compute bias correction
bias_correction1 = 1 - beta1**t
bias_correction2 = 1 - beta2**t

# Update first moment
m_state[i].mul_(beta1).add_(grad, alpha=1-beta1)

# Update second moment
v_state[i].mul_(beta2).add_(grad * grad, alpha=1-beta2)

# Correct bias
m_hat = m_state[i] / bias_correction1
v_hat = v_state[i] / bias_correction2

# Update parameters
param.addcdiv_(m_hat, torch.sqrt(v_hat) + eps, value=-lr)
```

**AdamW**:
```python
# Apply weight decay (decoupled)
param.mul_(1 - lr * weight_decay)

# Then proceed with standard Adam update
# [Same code as Adam above]
```

**Using Built-in Optimizers**:
```python
adagrad_opt = optim.Adagrad(model.parameters(), lr=0.01)
adam_opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
adamw_opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

## 5. Minimal Working Example

Can view the [colab here](https://colab.research.google.com/github/damek/STAT-4830/blob/main/section/9/notebook.ipynb)

The script

1. Defines a three-layer feedforward neural network
2. Loads and preprocesses the Fashion MNIST dataset
3. Trains identical model architectures with four different optimizers (SGD with momentum, AdaGrad, Adam, and AdamW)
4. Tracks training and validation metrics for each optimizer
5. Visualizes the results

The results are as follows:

![optimizer_comparison](figures/optimizer_comparison.png)
![optimizer_timing](figures/optimizer_timing.png)

The plots show that on this problem, all the methods are roughly comparable, with a slight advantage for the adaptive methods. It's incredibly difficult to compare optimization algorithms from an empirical perspective. The results depend on the model, the dataset, and the hyperparameters. Recently some researchers at Google, Meta, Dell, and in Academia have started the [algoperf benchmark](https://github.com/mlcommons/algorithmic-efficiency), which tests optimizers on a variety of medium scale machine learning tasks. They've also written a [paper](https://arxiv.org/abs/2306.07179) on the difficulty of benchmarking neural network training algorithms. We'll discuss some of these issues in a later lecture, but for now let this just serve as a warning that drawing conclusions from a few plots is hard!