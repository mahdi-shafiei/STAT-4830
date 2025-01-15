---
layout: slides
title: Linear Regression - Direct and Iterative Methods
---

# Linear Regression
## From Direct to Iterative Methods

---

# Motivating Example
- House price prediction
- Square footage → Price
- Find the best-fitting line

```python
# Code will be shown in notebook
```

---

# Direct Methods
- Normal equations
- One-shot solution
- Exact answer
- But: expensive for large datasets

---

# Scaling Challenges
- Matrix multiplication: O(n²)
- Matrix inversion: O(n³)
- Memory usage grows quadratically

---

# Gradient Descent
- Iterative approach
- Small steps downhill
- Memory efficient
- Works well at scale

---

# Feature Mappings
- Transform input space
- Fit non-linear patterns
- Same linear machinery
- Example: polynomial features 