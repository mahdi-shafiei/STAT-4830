# Rough Outline

## Linear regression and (stochastic) gradient descent
- The classic methods: high accuracy solutions, slowly
  - Gaussian elimination, LU, ...
- Modern iterative methods: modest accuracy solutions, quickly
  - Gradient descent: minimize linear expansion
  - Stochastic gradient methods: sample and minimize linear expansion
- Issues: stepsizes, batches, interpolation….

## Optimization formulations in data science, machine learning, and sequential decision making
### Statistical estimation and inverse problems in data science
- The set-up: inverting a nonlinear mapping
- Compressive sensing 
- Phase retrieval

### Prediction problems
- Models, loss functions, regularization
- Minimizing prediction error 
- Maximizing the likelihood of data
- Leveraging prior information
- Population vs empirical problems

### Sequential decision making problems
- LQR
- Bandits 
- Reinforcement learning

### Considerations
- Convexity, smoothness, stochasticity

### Classical convex formulations
- LP, QP, and SDP
- Software: CVXPY and others (e.g., https://developers.google.com/optimization/math_opt)

## How to think about calculus
- Gradients, Hessians, and Taylor's theorem
- Gradient: direction of steepest descent
- First and second order optimality conditions

### Linear algebra needed
- Linear transformations/inner products
- Norms
- Jacobians and the chain rule 
- Formal gradients of nondifferentiable functions

## You will never differentiate again
### Formal introduction to auto differentiation
- Software: 
  - The details: Micrograd (https://github.com/karpathy/micrograd)
- Intro to PyTorch (possibly Jax)
  - Architectures
  - Colab
  - GPUs

## (Stochastic) gradient descent
- Gradient descent
- Stochastic gradient descent
- Optimizing the population risk

### What to expect from theory
- Convexity: finds global minima, has rates, can be accelerated
- Nonconvexity: finds critical points, has rates, can be accelerated in certain cases
- In limited cases e.g., NTK, we can minimize the training loss

### Modifications
- Adagrad, Adam, momentum, Nesterov acceleration,...
- https://pytorch.org/docs/stable/optim.html#algorithms

### Issues that affect dynamics
- Stepsize schedule
  - Warm up, decay, cosine…
- Ill conditioning 
- Interpolation

### Online gradient descent and regret

## Beyond gradients: addressing ill conditioning 
- Newton's, Gauss-Newton and quasi Newton methods
- Other preconditioners
  - Natural gradient descent
  - KFAC
  - Shampoo (https://arxiv.org/pdf/2406.17748)

### Practical considerations: solving the linear system 
- Conjugate gradient
- GMRES
- CoLA software library (https://github.com/wilson-labs/cola)

## Tentative topics beyond the basics
- Zeroth order methods
- Mirror Descent
- Constraints and regularization 
- Proximal and projected gradient methods
- Examples: 
  - Compressive sensing, low-rank matrix completion
- Alternating minimization style methods
  - Examples
    - Layer wise training of deep networks
    - Sparse coding/dictionary learning
- Optimization over low-rank matrices and tensors 
  - Burer Monteiro and Gauss-newton
- Distributed data: federated learning
- Sensitive data: differentially private gradient methods
  - Basic principle: noise injection
- Curriculum learning
- Low precision optimizers

## Tuning deep learning performance
- Predictive models of performance: Mu_p and Scaling "laws"
- Deep learning tuning playbook (https://github.com/google-research/tuning_playbook)
- Benchmarking Neural Network Training Algorithms (https://arxiv.org/abs/2306.07179)

## Some empirical models
- Sampling text via transformers and MinGPT
- Sampling images with diffusion models

