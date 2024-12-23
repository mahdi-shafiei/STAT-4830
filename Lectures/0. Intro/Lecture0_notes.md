# Lecture 0: Intro to the course

Outline: 
* Content 
* Deliverables 
* Expectations and learning outcomes

## What is optimization? 

In the syllabus, I wrote: 

> Optimization is the modeling language in which modern data science, machine learning, and sequential decision-making problems are formulated and solved numerically. 

Let's unpack this. 

Let's say we have a problem we want to solve or a goal we want to achieve in the real world. We want to do so in the "best" way possible while respecting some constraints. 

Translating this into an optimization problem requires us to have thought clearly enough about our goals that we can answer the following questions:
* How can we encode the decision we wish to make or strategy we wish to take (or an approximation thereof) as a list of numbers (e.g., a vector).
* How can we formulate a metric that measures how close we are to the goal or solution?
* How can we formulate the constraints that bound the actions we can take?

Going from a real-world problem to an optimization formulation is the *modeling processing.* It often requires multiple iterations and communication with domain experts. It is more art than science. 

The application areas of this course mostly have preexisting mathematical formulations that you can start from. 

In this course, we will focus on three application areas: 
1. Data science (e.g., statistical estimation, inverse problems)
2. Machine learning (e.g., predictive modeling, generative modeling)
3. Sequential decision-making (e.g., control, bandits, reinforcement learning)


<!-- 
> This course will teach you how to formulate these problems mathematically, choose appropriate algorithms to solve them, and implement and tune the algorithms in PyTorch. Tentative topics include:

* Optimization-based formulations of statistical estimation and inverse problems in data science; predictive and generative models in machine learning; and control, bandit, and reinforcement learning problems in sequential decision-making. 

* A high-level tour of the foundations of mathematical optimization, viewed as an algorithmic discipline, and what to expect from theory; key considerations such as convexity, smoothness, saddle points, and stochasticity; classical formulations, such as linear, quadratic, and semidefinite programs; numerical solvers such as CVXPY.

* Popular optimization methods such as (online and stochastic) gradient methods, (quasi) Newton methods, algorithmic extensions to constrained, regularized, and distributed problems, as well as optimization methods that preserve privacy of sensitive data. 

* Modern software libraries such as PyTorch and Jax and the principles underlying "automatic differentiation" techniques. Best practices in tuning optimization methods, e.g., in deep learning problems.

By the end of this course, you will become an intelligent consumer of numerical methods and software for solving modern optimization problems.  -->