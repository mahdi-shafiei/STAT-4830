# Lecture 0: Intro to the Course

**Outline**  
- Content  
- Deliverables  
- Expectations and learning outcomes  

---

## What Is Optimization?

In the syllabus, I mentioned:

> Optimization is the modeling language in which modern data science, machine learning, and sequential decision-making problems are formulated and solved numerically.

Let's unpack what that means. Suppose we have a real-world problem or a goal we want to achieve in the best possible way, subject to certain constraints. To do this, we have to:

1. **Represent Our Decisions**  
   We need to encode our decision or strategy (or an approximation of it) as a list of numbers (e.g., a vector). For instance, if we're allocating resources between two projects, we might define $x = (x_1, x_2)$, where $x_1$ is the budget for project 1 and $x_2$ for project 2.

2. **Define an Objective Function**  
   We then formulate a metric to measure how close we are to achieving our goal. If the goal is to minimize cost or error, we could write $\min_x f(x)$. For example, we might minimize
   $$f(x) = (x_1 - 100)^2 + (x_2 - 50)^2$$
   to keep the budgets near some target values.

3. **Identify Relevant Constraints**  
   Finally, we specify any bounds or conditions that our choices must satisfy, such as $x_1 + x_2 \leq 150$ to keep total spending below 150, or $x_1, x_2 \ge 0$ to ensure the budgets are nonnegative.

This process—called *optimization modeling*—often involves creativity and iteration. Real-world complexity requires you to rethink which objectives and constraints really matter, and you might consult domain experts to confirm whether your formulation makes sense.

---

## Application Areas and Their Optimization Goals

Here, we'll look at three major application areas: Data Science, Machine Learning, and Sequential Decision-Making. Although each area uses optimization in a slightly different way, they all rely on choosing "best" parameters under certain objectives and constraints.

### 1. Data Science (e.g., Statistical Estimation, Inverse Problems)

Here the key idea is to estimate unknown parameters from noisy or incomplete data. For example, imagine a doctor reconstructing a 3D image from 2D X-ray slices. The main optimization formulation here tries to balance how well the model fits the observed data (such as a least-squares term) with various physical or regularization constraints (like smoothness or nonnegativity). The constraints we enforce often reflect real-world conditions, such as parameter ranges or physical laws that must be respected.

### 2. Machine Learning (Predictive and Generative Modeling)

Here the key idea is to learn a function or distribution from data. For a predictive model (like a classifier), the main objective is usually to minimize some loss function, such as classification error. For a generative model (like a Large Language Model), we aim to maximize the likelihood of observed data—for instance, by predicting the next token in a sequence. The constraints we enforce might include regularization or structural limits on the model architecture, ensuring it generalizes and remains computationally manageable.

### 3. Sequential Decision-Making (e.g., Control, Bandits, Reinforcement Learning)

Here the key idea is to make a series of decisions over time, where each action influences future states and rewards. A classic example is a robot learning to walk by maximizing cumulative reward (distance walked without falling), subject to torque limits and physical laws. The main optimization formulation here tries to account for long-term consequences, often requiring techniques like dynamic programming or policy gradients. The constraints we enforce can involve real-time performance, safety boundaries, or resource limitations.

---

Despite their differences, these application areas all involve finding "best" parameters—whether "best" means most accurate predictions, most coherent outputs, or most effective long-term strategies. The art is in how you formulate your goals and constraints so that an algorithm can meaningfully address the problem.


<!-- 
> This course will teach you how to formulate these problems mathematically, choose appropriate algorithms to solve them, and implement and tune the algorithms in PyTorch. Tentative topics include:

* Optimization-based formulations of statistical estimation and inverse problems in data science; predictive and generative models in machine learning; and control, bandit, and reinforcement learning problems in sequential decision-making. 

* A high-level tour of the foundations of mathematical optimization, viewed as an algorithmic discipline, and what to expect from theory; key considerations such as convexity, smoothness, saddle points, and stochasticity; classical formulations, such as linear, quadratic, and semidefinite programs; numerical solvers such as CVXPY.

* Popular optimization methods such as (online and stochastic) gradient methods, (quasi) Newton methods, algorithmic extensions to constrained, regularized, and distributed problems, as well as optimization methods that preserve privacy of sensitive data. 

* Modern software libraries such as PyTorch and Jax and the principles underlying "automatic differentiation" techniques. Best practices in tuning optimization methods, e.g., in deep learning problems.

By the end of this course, you will become an intelligent consumer of numerical methods and software for solving modern optimization problems.  -->