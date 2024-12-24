# Lecture 0: Intro to the Course

**Outline**  
- Content  
- Deliverables  
- Expectations and learning outcomes  

## What Is Optimization?

In the syllabus, I mentioned:

> Optimization is the modeling language in which modern data science, machine learning, and sequential decision-making problems are formulated and solved numerically.

Let's unpack what that means. Suppose we have a real-world problem or a goal we want to achieve in the best possible way, subject to certain constraints. To do this, we have to:

1. **Represent Our Decisions**  
   We need to encode our decision or strategy (or an approximation of it) as a list of numbers (e.g., a vector). For instance, if we're allocating resources between two projects, we might define $x = (x_1, x_2)$, where $x_1$ is the budget for project 1 and $x_2$ for project 2.

2. **Define an Objective Function**  
   We then formulate a metric to measure how close we are to achieving our goal. If the goal is to minimize cost or error, we could write $\min_x f(x)$. For example, we might minimize

   $$
   f(x) = (x_1 - 100)^2 + (x_2 - 50)^2
   $$
   to keep the budgets near some target values.

3. **Identify Relevant Constraints**  
   Finally, we specify any bounds or conditions that our choices must satisfy, such as $x_1 + x_2 \leq 150$ to keep total spending below 150, or $x_1, x_2 \ge 0$ to ensure the budgets are nonnegative.

This process—called *optimization modeling*—often involves creativity and iteration. Real-world complexity requires you to rethink which objectives and constraints really matter, and you might consult domain experts to confirm whether your formulation makes sense.

## Application Areas and Their Optimization Goals

Below are three major application areas where we'll spend most of our time. Although each area uses optimization in a slightly different way, they all rely on choosing “best” parameters under certain objectives and constraints.

### 1. Data Science (e.g., Statistical Estimation, Inverse Problems)

Here the key idea is to estimate unknown parameters from noisy or incomplete data. For example, imagine a doctor reconstructing a 3D image from 2D X-ray slices. The main optimization formulation here tries to balance how well the model fits the observed data (such as a least-squares term) with various physical or regularization constraints (like smoothness or nonnegativity). 

### 2. Machine Learning (Predictive and Generative Modeling)

Here the key idea is to learn a function or distribution from data. For a predictive model (like a classifier), the main objective is to minimize a loss function, such as classification error. For a generative model (like a Large Language Model), we aim to maximize the likelihood of observed data—for instance, by predicting the next token in a sequence. The constraints might include regularization or structural limits on the model architecture.

### 3. Sequential Decision-Making (e.g., Control, Bandits, Reinforcement Learning)

Here the key idea is to make a series of decisions over time, where each action influences future states and rewards. A classic example is a robot learning to walk by maximizing cumulative reward (distance traveled without falling). The main optimization formulation here tries to account for long-term consequences, often requiring techniques like dynamic programming or policy gradients. Constraints can involve real-time performance, safety boundaries, or resource limitations.

---

Despite their differences, these application areas all involve finding "best" parameters—whether "best" means most accurate predictions, most coherent outputs, or most effective long-term strategies. The art is in how you formulate your goals and constraints so that an algorithm can meaningfully address the problem.


## Content

In this course, we will study the mathematical foundations of optimization while focusing on how to implement methods for real problems. Below is a roadmap of how we will approach these topics: we begin with a simple setting (linear regression) and a basic algorithm (gradient descent), then expand to broader data science, machine learning, and sequential decision-making applications. Throughout, we will highlight the necessary calculus and linear algebra, practical implementation details, and strategies for large-scale problems.

1. **Linear Regression and (Stochastic) Gradient Descent**  
   We start with linear regression and examine direct methods such as LU factorization and Gaussian elimination. These methods work well for moderate problem sizes, but they struggle with very large datasets. This limitation naturally leads to iterative methods like gradient descent and stochastic gradient descent. We will compare their strengths, discuss stepsizes and mini-batch sizes, and learn when “modest accuracy” can be more valuable than “perfect precision.”

2. **Optimization Formulations in Data Science, Machine Learning, and Sequential Decision Making**  
   Many real-world problems can be cast as choosing parameters to minimize or maximize a well-defined objective, often under constraints. In this section, we will learn how to set up these formulations in different contexts:

   - **Statistical Estimation and Inverse Problems in Data Science**  
     A common theme here is *inverting a nonlinear mapping* from measurements back to the underlying parameters. Classic examples include:
     - **Compressive Sensing**: Recovering a high-dimensional signal from a small number of observations by leveraging sparsity or other structural properties.  
     - **Phase Retrieval**: Reconstructing signals from the magnitude (but not the phase) of certain transforms.  
     These problems often involve constraints or regularization to account for noise or missing data, and they can be framed in a variety of convex or nonconvex ways.

   - **Prediction Problems**  
     In machine learning, we often aim to *minimize a loss function* or *maximize the likelihood* of observed data. We will explore how models, loss functions, and regularizers come together to prevent overfitting and leverage prior information. A key distinction is between:
     - **Population Problems**: Where the goal is to do well on the true data-generating process (often unknown).  
     - **Empirical Problems**: Where the objective is computed on a finite sample.  
     We will see how ideas of convexity, smoothness, and stochasticity influence which algorithms and analyses are most appropriate.

   - **Sequential Decision Making**  
     Sometimes we need to plan multiple steps ahead, where each decision affects future options. We will look at:
     - **LQR (Linear Quadratic Regulator)**: A control problem that can often be solved via convex or dynamic programming methods.  
     - **Bandits**: Settings where decisions are made under uncertainty about rewards, balancing exploration and exploitation.  
     - **Reinforcement Learning**: More general environments where actions, states, and rewards evolve over time.  
     These problems highlight the role of *stochasticity* and can involve large-scale optimization with sparse feedback.

   - **Classical Convex Formulations and Software Tools**  
     Throughout these examples, we will see that many subproblems fit classical templates like Linear Programs (LP), Quadratic Programs (QP), or Semidefinite Programs (SDP). We will use software tools like [CVXPY](https://www.cvxpy.org) and [Google’s Math Opt](https://developers.google.com/optimization/math_opt) to solve these formulations directly, gaining insight into both their theory and their practical performance.

    By examining these diverse examples, we will discover common threads—such as the importance of convexity and the challenges of dealing with nonconvex or noisy objectives—and develop a systematic way of approaching optimization in modern data science and machine learning pipelines.

3. **How to Think About Calculus**  
   We will review gradients, Hessians, and Taylor expansions, and see how they are used guide optimization algorithms. We will introduce relevant linear algebra will include inner products, matrix decompositions, and norms—essential mathematical background when dealing with high-dimensional spaces.

4. **Automatic Differentiation**  
   Automatic differentiation (AD) is a core driver of the modern deep learning revolution. Frameworks like PyTorch and Jax handle derivatives under the hood, letting us iterate quickly on new models without having to manually compute gradients. These frameworks have made it feasible rapidly iterate on machine learning code, allowing one to build large neural networks, experiment with architectures, and tune every aspect of the model without the overhead of recomputing gradients after each change. We will explain how autodiff works, focusing first on a minimal library (Micrograd). Then we will move on to modern libraries like PyTorch and Jax to show how they scale to more complex tasks.


5. **(Stochastic) Gradient Descent**  
   We take a deeper look at gradient-based methods in the context of large datasets, where computing the gradient on every data point is too expensive. Stochastic and mini-batch variants of gradient descent help by sampling subsets of data, offering faster updates and the ability to converge to good solutions in practice. We will distinguish between optimizing the *population* risk (the true underlying objective) versus the *empirical* risk (the objective computed on a finite dataset). We’ll examine what theory can tell us about the guarantees of these methods: for convex problems, they can find global minima and achieve provable rates of convergence; for nonconvex problems, they typically find critical points but can still converge quickly under certain assumptions (e.g., in the neural tangent kernel regime, it may be possible to minimize the training loss).

   Building on the basic stochastic gradient method, we will explore major algorithmic extensions, such as Adagrad, Adam, momentum, and Nesterov acceleration—tools that appear in PyTorch and other deep learning frameworks as standard choices. We will see how stepsize schedules (including warm-up, decay, and cosine schedules), ill-conditioning, and interpolation effects can change the dynamics of optimization. Throughout, the focus will be on understanding both the theoretical trade-offs and practical considerations that arise when training models with large, complex parameter spaces.


6. **Beyond Gradients: Addressing Ill Conditioning**  
   Some optimization landscapes are harder to navigate with only first-order information. Here we look at second-order methods (Newton, Gauss-Newton, quasi-Newton) and preconditioners that reshape the problem for faster convergence. We will also discuss linear solvers (Conjugate Gradient, CoLA) that make these approaches more practical for large problems.

7. **Tentative Topics Beyond the Basics**  
   Depending on time and student interests, we may explore:
   - **Zeroth-Order Methods**: Situations where gradients are unavailable or expensive.  
   - **Constraints and Regularization**: Projections and proximal operators for nonsmooth or constrained objectives.  
   - **Distributed and Federated Optimization**: Large-scale settings where data is spread across multiple machines.  
   - **Privacy-Preserving Methods**: Adjusting updates for differential privacy.

8. **Tuning Deep Learning Performance**  
   Many problems involve large neural networks. We will discuss scaling laws (e.g., Mu_p), strategies from a “Deep Learning Tuning Playbook" (by Google researchers), and benchmark efforts that measure and compare different optimization algorithms.

9. **Empirical Models**  
   We close with examples of state-of-the-art generative models:
   - **Transformers and GPT-like Models**: Text generation through sequential prediction.  
   - **Diffusion Models**: Image generation or restoration via iterative denoising.  

The details of this outline shift based on class interests. By the end of the course, you will have a toolbox of optimization methods, an understanding of their theoretical underpinnings, and practical experience in applying them to real problems.


## Deliverables

Your main deliverables in this course will be:

- **Homework (20%)**  
  Two (tentative) homework assignments focusing on theory and implementation.

- **Quizzes (20%)**  
  Short, regular quizzes to keep you on track with the material.

- **Course Project (60%)**  
  A substantial group project where you apply the optimization ideas we discuss in class. You’ll propose a project, demonstrate progress midway, and deliver a final write-up with code. Expect to use Python notebooks with PyTorch or other libraries. 
  - (10%) Initial proposal  
  - (10%) Midterm presentation  
  - (40%) Final report, presentation, and software deliverable  

Projects can include training/fine-tuning models, reproducing optimization results, or benchmarking new algorithms on public datasets. You’re encouraged to apply these ideas to research problems or topics that spark your curiosity.

---

## Expectations and Learning Outcomes

1. **Modeling and Formulation**  
   By the end of this course, you should be able to take a real-world problem (in data science, machine learning, or sequential decision-making) and translate it into a formal optimization problem with well-defined objectives and constraints.

2. **Algorithm Selection and Analysis**  
   You will learn how to choose an appropriate algorithm—from basic gradient descent to more advanced quasi-Newton methods—and understand the trade-offs of different approaches (speed, accuracy, scalability).

3. **Software Proficiency**  
   We will use modern libraries like PyTorch (possibly Jax) to implement and experiment with these algorithms. You will gain experience with auto differentiation and learn best practices for tuning and debugging iterative solvers.

4. **Practical Sensibility**  
   Although we’ll cover fundamental optimization theory (convexity, convergence rates, saddle points, etc.), the focus is on practical usage. You will learn which methods to try first and how to iterate quickly when working with large datasets and complicated models.

5. **Research-Level Thinking**  
   This course also prepares you for research or advanced development tasks. You’ll see how to benchmark optimization methods, reproduce existing studies, and innovate new ways to handle constraints like privacy and distributed data.

Ultimately, you will gain the skills to set up and solve optimization problems, knowing when to rely on standard libraries and when to use specialized methods. These skills will be valuable whether you’re coding models for industrial machine learning pipelines or doing academic research.



<!-- 
> This course will teach you how to formulate these problems mathematically, choose appropriate algorithms to solve them, and implement and tune the algorithms in PyTorch. Tentative topics include:

* Optimization-based formulations of statistical estimation and inverse problems in data science; predictive and generative models in machine learning; and control, bandit, and reinforcement learning problems in sequential decision-making. 

* A high-level tour of the foundations of mathematical optimization, viewed as an algorithmic discipline, and what to expect from theory; key considerations such as convexity, smoothness, saddle points, and stochasticity; classical formulations, such as linear, quadratic, and semidefinite programs; numerical solvers such as CVXPY.

* Popular optimization methods such as (online and stochastic) gradient methods, (quasi) Newton methods, algorithmic extensions to constrained, regularized, and distributed problems, as well as optimization methods that preserve privacy of sensitive data. 

* Modern software libraries such as PyTorch and Jax and the principles underlying "automatic differentiation" techniques. Best practices in tuning optimization methods, e.g., in deep learning problems.

By the end of this course, you will become an intelligent consumer of numerical methods and software for solving modern optimization problems.  -->