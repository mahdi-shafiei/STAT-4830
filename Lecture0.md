# Lecture 0: Intro to the Course

## Table of Contents
1. [Course syllabus and key points](#course-syllabus-and-key-points)
2. [Preview: Solving spam classification with optimization](#preview-solving-spam-classification-with-optimization)
3. [Tentative course structure](#tentative-course-structure)
4. [Expectations and learning outcomes](#expectations-and-learning-outcomes)

### Notebooks

[Colab notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/Lecture0.ipynb) for the spam filter example.

## Course Syllabus and key points
Welcome to STAT 4830: Numerical optimization for data science and machine learning. This course teaches you how to formulate optimization problems, select and implement algorithms, and use frameworks like PyTorch to build and train models. Below are some highlights of the syllabus to get you oriented:

### Prerequisites

- Basic calculus and linear algebra (Math 2400).
- Basic probability (Stat 4300).
- Familiarity with Python programming.
- You do not need a background in advanced optimization or machine learning research. We’ll cover the fundamentals together.

### Schedule and Format

- This is primarily a lecture-based course with weekly programming exercises.
- We will introduce theory (convexity, gradient-based methods, constraints, etc.) and then apply it in Python notebooks using PyTorch (and occasionally other libraries like CVXPY).

### Deliverables

- A single final project that you begin drafting by Week 2 and refine throughout the semester.
- Several “checkpoints” (drafts, presentations) so you can get feedback and improve incrementally.
- The final submission will consist of a GitHub repository (code, report, slides) plus a polished demonstration (e.g., a Google Colab notebook).

### Why PyTorch?

- We are focusing on PyTorch because deep learning’s success has been driven in part by modern auto-differentiation frameworks.
- These frameworks allow for rapid experimentation with new model architectures and optimization algorithms—something that older solver-based tools (like CVX or early MATLAB packages) did not fully accommodate.

### Who Is This Course For?

- Targeted at junior/senior undergrads, but also valuable for PhD students wanting to incorporate numerical optimization into their research. Students who have met the prerequisites are welcome to join.
- If you already have a research project that involves model fitting or data analysis, this course may deepen your toolkit and sharpen your understanding of optimization.
- We will keep refining the course content based on your interests. If you have a particular topic, domain, or application you’d like to see, let me know.


## Preview: Solving spam classification with optimization

Let's start with a classic problem: sorting important emails from spam. 

### How computers read email

Consider two messages that land in your inbox:

```python
email1 = """
Subject: URGENT! You've won $1,000,000!!!
Dear Friend! Act NOW to claim your PRIZE money!!!
Click here: www.totally-legit-prizes.com
"""

email2 = """
Subject: Team meeting tomorrow
Hi everyone, Just a reminder about our 2pm project sync.
Please review the attached agenda.
"""
```

Computers can't directly understand text like we do. Instead, we convert each email into numbers called features. Think of features as measurements that help distinguish spam from real mail:

```python
def extract_features(email):
    features = {
        'exclamation_count': email.count('!'),
        'urgent_words': len(['urgent', 'act now', 'prize'] & set(email.lower().split())),
        'suspicious_links': len([link for link in email.split() if 'www' in link]), # Any link is suspicious imo
        'time_sent': email.timestamp.hour,  # Spam often sent at odd hours
        'length': len(email)
    }
    return features

# Our spam email gets turned into numbers
spam_features = extract_features(email1)
print(spam_features)
# {'exclamation_count': 8,
#  'urgent_words': 3,
#  'suspicious_links': 1,
#  'time_sent': 3,
#  'length': 142}
```

### From features to decisions

How do we decide if an email is spam? We assign a weight to each feature. Think of weights as measuring how suspicious each feature is:

```python
import torch

weights = {
    'exclamation_count': 0.5,    # More ! marks → more suspicious
    'urgent_words': 0.8,         # "Urgent" language → very suspicious
    'suspicious_links': 0.6,     # Unknown links → somewhat suspicious
    'time_sent': 0.1,           # Time matters less
    'length': -0.2              # Longer emails might be less suspicious
}

# PyTorch needs numbers in tensor form
features = torch.tensor([1.0, 3.0, 1.0, 3.0, 142.0])
w = torch.tensor(list(weights.values()), requires_grad=True)
```

### The classification process

![Spam Classification Process](Lecture%20resources/Lecture%200/figures/spam_classification_process.png)

Features flow through a sequence of transformations:
1. Extract numeric features from raw text
2. Multiply each feature by its weight
3. Sum up the weighted features
4. Convert the sum to a probability using the sigmoid function

### Making the decision: the sigmoid function

We combine features and weights to get a "spam score". But how do we turn this score into a yes/no decision? We use a function called sigmoid that turns any number into a "probability" between 0 and 1:

![Sigmoid Function](Lecture%20resources/Lecture%200/figures/sigmoid.png)

```python
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def spam_score(features, weights):
    raw_score = torch.dot(features, weights)  # Combine features and weights
    probability = sigmoid(raw_score)          # Convert to probability
    return probability
```

The sigmoid function provides crucial properties:
- Very negative scores → probabilities near 0 (definitely not spam)
- Very positive scores → probabilities near 1 (definitely spam)
- Zero score → probability of 0.5 (maximum uncertainty)

### The mathematical problem: finding optimal weights

Our spam filter needs to find weights that correctly classify emails. We can write this as an optimization problem:

$$
\min_{w} \frac{1}{n} \sum_{i=1}^n \left[ -y_i \log(\sigma(x_i^\top w)) - (1-y_i) \log(1-\sigma(x_i^\top w)) \right]
$$

Where:
- $w$ are the weights we're trying to find (a vector with 5 entries)
- $x_i$ are the features of email $i$ (another vector with 5 entries)
- $x_i^\top w$ is the dot product of $x_i$ and $w$ (a scalar)
- $y_i$ is $1$ if email $i$ is spam, $0$ if not
- $\sigma$ is the sigmoid function 
- $n$ is the number of training emails

This formula measures our mistakes (called "cross-entropy loss").

### Why cross-entropy loss works

The cross-entropy loss teaches our model to make confident, correct predictions while severely punishing mistakes. Let's see how it works by examining the two curves in our plot, which show how we penalize predictions for spam and non-spam emails.

![Cross-Entropy Loss](Lecture%20resources/Lecture%200/figures/cross_entropy.png)
<!-- [Figure: Cross-Entropy Loss - See cross_entropy.tex] -->

Consider a legitimate email (not spam, label = 0). Our model assigns it a probability of being spam:

```python
uncertain_prediction = 0.3   # "Maybe spam (30% chance)?"
confident_wrong = 0.8       # "Probably spam (80% chance)"
confident_right = 0.1       # "Probably not spam (10% chance)"
```

The green curve ("Wrong: Not Spam") shows how we penalize mistakes on legitimate emails:
- An uncertain wrong prediction (0.3) receives a moderate penalty
- A confident wrong prediction (0.8) triggers a severe penalty
- The penalty approaches infinity as the prediction approaches 1.0

The red curve ("Right: Spam") works similarly but for actual spam emails. Together, the curves create a powerful learning dynamic:

1. **Uncertainty gets corrected**: The point "Uncertain & Wrong" shows a prediction hovering around 0.3 - not catastrophically wrong, but penalized enough to encourage learning.

2. **Confidence gets rewarded**: The point "Confident & Right" shows a prediction around 0.1, receiving minimal penalty. This reinforces correct, confident predictions.

3. **Catastrophic mistakes get prevented**: Both curves shoot upward at their edges, creating enormous penalties for high-confidence mistakes. This prevents the model from becoming overly confident in the wrong direction.

The two curves meet at 0.5 (our decision boundary), creating perfect symmetry between spam and non-spam mistakes. This balance means our model:
- Learns equally from both types of examples
- Develops appropriate uncertainty when evidence is weak
- Gains confidence only with strong supporting evidence

The mathematics behind this intuition uses the logarithm:
```python
# For legitimate email (label = 0):
loss = -log(1 - prediction)   # Penalize high predictions

# For spam email (label = 1):
loss = -log(prediction)       # Penalize low predictions
```

Through training, these penalties guide the model from uncertainty toward confident, correct predictions:
```python
# Before training (uncertain)
email1 = 0.48  # Struggles to classify
email2 = 0.51  # Nearly random guesses

# After training (confident + correct)
email1 = 0.02  # Confidently marks non-spam
email2 = 0.97  # Confidently marks spam
```

Each training step pushes predictions away from the uncertain middle and toward confident extremes - but only when the evidence justifies that confidence. This careful balance between confidence and caution produces a robust spam filter that makes reliable decisions.

### How the optimization process works: following the gradient

Imagine you're hiking in a valley and want to reach the lowest point. A natural strategy is to:

1. Look at the ground around you
2. Take a step in the steepest downhill direction
3. Repeat until you can't go lower

![Gradient descent visualization showing path from high point to minimum](Lecture%20resources/Lecture%200/figures/gradient_descent.png)
This is exactly how the most well-known algorithm for optimization--called gradient descent--works. 

### Finding the weights with PyTorch

We can implement the gradient descent algorithm in PyTorch. Here's how each step or iteration works:

1. Measure how each weight affects our mistakes
2. Adjust weights to reduce future mistakes
3. Get closer to weights that separate spam from legitimate email

The process repeats until the mistakes can't be reduced further: 

```python
# Start with random weights
weights = torch.randn(5, requires_grad=True)
learning_rate = 0.01

for _ in range(1000):
    # 1. Make predictions and calculate mistakes
    predictions = spam_score(features, weights)
    loss = cross_entropy_loss(predictions, true_labels)
    
    # 2. Calculate gradient (steepest direction)
    loss.backward()
    
    # 3. Take a step downhill
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        weights.grad.zero_()
```

More formally, each iteration measures how well our current weights classify *all* of our training emails, calculates a *gradient* of the loss function with respect the weights $w$ (an "error-reducing" direction), and updates all weights by taking a small step in this error-reducing direction. The learning rate (0.01) controls these steps - smaller values move more reliably toward better weights but take longer, while larger values move faster but risk overshooting good solutions.

The optimization process continues until it takes 1000 steps in the gradient direction. This is just one way to stop the algorithm; others exist. For example, we could instead halt when the loss plateaus (stops decreasing significantly) or reaches a target threshold. Each approach balances computation time against solution quality.

### Numerical results
When you checkout the [notebook](https://colab.research.google.com/github/damek/STAT-4830/blob/main/Lecture0.ipynb) for this lecture, this is what you'll see as you run the training loop: 
![Loss curves](Lecture%20resources/Lecture%200/figures/training_run.png)
The first plot shows the value of the cross-entropy loss as we train the model. This and the "training accuracy" (shown in the second plot) are both metrics that measure how well our model performs on the training data. They are are diagnostic plots -- they say at least the optimization part of the code is working. 

The third plot is far more important. It shows the performance of the model on a test set of emails that were not used to train the model.  A gap between training and test accuracy would signal overfitting, but here both metrics converge and stay stable, indicating the model generalizes well to new examples.

Together, these plots confirm our model learns a robust set of rules for distinguishing spam from legitimate email. The rapid initial progress followed by stability suggests the feature set captures the essential patterns in our data.

### The what, how, and why of PyTorch

PyTorch makes this optimization process painless by automating the most challenging part - computing gradients. As calculations flow through our code, PyTorch builds a record of operations (a computational graph). When we call `backward()`, this graph enables automatic calculation of all required derivatives.

The framework achieves this automation through tensors - its fundamental building blocks that store numbers in grid structures representing single values, lists, tables, or higher-dimensional arrays. PyTorch uses tensors because they track operation history for automatic gradients and enable parallel computation. This design lets code run efficiently on both CPUs and GPUs with minimal changes.

We'll talk more about these intricacies as the course progresses.

### What you'll learn in this course

While our spam filter demonstrates these concepts simply, the same pattern powers most machine learning: predict outputs, measure errors, compute gradients, adjust parameters. Different problems require different loss functions or model architectures, but this optimization loop remains central and is the focus of this course.

Our spam filter illustrates the key ideas you'll learn in this course:
1. Converting real problems into optimization problems
2. Choosing appropriate optimization methods
3. Implementing solutions in PyTorch

## Tentative Course Structure

So what will we cover in this course?

### 1. Linear Algebra, Regression, and Direct Methods
We begin with the essential tools: norms, inner products, and matrix decompositions. Linear regression serves as our first optimization problem, solvable through direct methods like LU factorization and Gaussian elimination. These methods work well for moderate problem sizes but struggle with large datasets, motivating our transition to iterative methods.

### 2. Problem Formulations and Classical Software

This section introduces optimization formulations in several core application areas, and classical formulations that are solveable by existing software tools.

First, we explore problem formulations in three core application areas:

**Statistical Estimation and Inverse Problems**
Reconstruct signals from indirect observations - as in the reconstruction of the first black hole image from radio telescope data.

**Machine Learning Models**
Train models that learn patterns from data, whether for prediction (spam detection) or generation (language models).

**Sequential Decision Making**
Make decisions in sequence where each choice reshapes future possibilities - from chess strategies to spacecraft navigation.

Second, we study classical **convex formulations**, such as Linear Programs (LP), Quadratic Programs (QP), and Semidefinite Programs (SDP). These formulations have well-understood properties and moderately sized problem can be solved efficiently by existing software tools, such as CVXPY and others (e.g., [Google's MathOpt](https://developers.google.com/optimization/math_opt)). While our core applications often involve more complex optimization problems, recognizing when parts of them fit into these classical templates allows us to leverage powerful, ready-made solvers.

### 3. Calculus for Optimization
Gradients, Hessians, and Taylor expansions provide the mathematical foundation for optimization algorithms. This section explains how these tools guide algorithm design and implementation.

### 4. Automatic Differentiation and PyTorch
Automatic differentiation (AD) powers modern deep learning frameworks. We examine how AD works, starting with a minimal implementation (Micrograd) before moving to PyTorch. This progression reveals how frameworks handle derivatives automatically, enabling rapid iteration on complex models.

### 5. First-Order Methods: (Stochastic) Gradient Descent
Large datasets make computing exact gradients impractical. Stochastic and mini-batch variants of gradient descent offer a solution by sampling data subsets. We examine theoretical guarantees (global minima for convex problems, critical points for nonconvex ones) and practical modifications like Adam, momentum, and learning rate schedules.

### 6. Second-Order Methods
Some optimization landscapes require more than gradient information. Newton, Gauss-Newton, and quasi-Newton methods reshape the optimization landscape for faster convergence. Linear solvers (Conjugate Gradient, CoLA) make these approaches practical for large problems.

### 7. Advanced Topics
Based on class interests, we may cover:
- Zeroth-order methods for settings without gradients
- Constrained optimization via projections and proximal operators
- Distributed optimization for data across multiple machines
- Privacy-preserving methods using differential privacy

### 8. Modern Practice in Deep Learning
We conclude with practical strategies for large-scale optimization:
- Scaling laws and performance prediction (Mu_p)
- Implementation strategies from Google's Deep Learning Tuning Playbook
- Benchmarking frameworks for comparing optimization algorithms
- Case studies in text generation (Transformers) and image generation (Diffusion Models)

The details of this outline shift based on class interests. By the end of the course, you will have a toolbox of optimization methods, an understanding of their theoretical underpinnings, and practical experience in applying them to real problems.

## Expectations and Learning Outcomes

What will you learn by the end of this course?

1. **Modeling and Formulation**  
   By the end of this course, you should be able to take a real-world problem (in data science, machine learning, or sequential decision-making) and translate it into a formal optimization problem with well-defined objectives and constraints.

2. **Algorithm Selection and Analysis**  
   You will learn how to choose an appropriate algorithm—from basic gradient descent to more advanced quasi-Newton methods—and understand the trade-offs of different approaches (speed, accuracy, scalability).

3. **Software Proficiency**  
   We will use modern libraries like PyTorch (possibly Jax) to implement and experiment with these algorithms. You will gain experience with auto differentiation and learn best practices for tuning and debugging iterative solvers.

4. **Optimization in Practice**  
   Although we’ll cover fundamental optimization theory (convexity, convergence rates, saddle points, etc.), the focus is on practical usage. You will learn which methods to try first and how to iterate quickly when working with large datasets and complicated models.

5. **Research Methods**  
   This course also prepares you for research or advanced development tasks. You’ll see how to benchmark optimization methods, reproduce existing studies, and innovate ways to handle constraints like privacy and distributed data.

When you finish, you’ll be equipped to handle the optimization component of modern data science and machine learning projects, appreciating both the theoretical and practical dimensions.




<!-- 
> This course will teach you how to formulate these problems mathematically, choose appropriate algorithms to solve them, and implement and tune the algorithms in PyTorch. Tentative topics include:

* Optimization-based formulations of statistical estimation and inverse problems in data science; predictive and generative models in machine learning; and control, bandit, and reinforcement learning problems in sequential decision-making. 

* A high-level tour of the foundations of mathematical optimization, viewed as an algorithmic discipline, and what to expect from theory; key considerations such as convexity, smoothness, saddle points, and stochasticity; classical formulations, such as linear, quadratic, and semidefinite programs; numerical solvers such as CVXPY.

* Popular optimization methods such as (online and stochastic) gradient methods, (quasi) Newton methods, algorithmic extensions to constrained, regularized, and distributed problems, as well as optimization methods that preserve privacy of sensitive data. 

* Modern software libraries such as PyTorch and Jax and the principles underlying "automatic differentiation" techniques. Best practices in tuning optimization methods, e.g., in deep learning problems.

By the end of this course, you will become an intelligent consumer of numerical methods and software for solving modern optimization problems.  -->