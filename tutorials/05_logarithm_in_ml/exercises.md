# Tutorial 05: Logarithm in ML — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Log Properties
Prove these fundamental properties:
a) $\log(ab) = \log(a) + \log(b)$
b) $\log(a^n) = n\log(a)$
c) $\log_a(b) = \frac{\log_c(b)}{\log_c(a)}$ (change of base)

### Exercise A2 🟢 — Why Log-Likelihood?
Explain why we maximize $\sum_i \log p(x_i|\theta)$ instead of $\prod_i p(x_i|\theta)$.

### Exercise A3 🟡 — Gradient Comparison
For likelihood $L(\theta) = \prod_{i=1}^n p(x_i|\theta)$:
a) Derive $\frac{\partial L}{\partial \theta}$
b) Derive $\frac{\partial \log L}{\partial \theta}$
c) Show that zeros occur at the same $\theta$

### Exercise A4 🟡 — Shannon's Uniqueness
Sketch the proof that the only function $I(p)$ satisfying:
1. $I(p) \geq 0$
2. $I(p) = 0$ only when $p = 1$
3. $I(p_1 p_2) = I(p_1) + I(p_2)$ (additivity)

...is $I(p) = -k\log(p)$ for some $k > 0$.

### Exercise A5 🔴 — Numerical Stability Analysis
Consider computing $\log(\sum_i e^{x_i})$ (log-sum-exp).
a) Show this can overflow/underflow
b) Derive the stable formula using $\max(x_i)$

### Exercise A6 🔴 — Is Log Fundamental?
**Thought experiment**: If computers had infinite precision, would log still be necessary? Argue mathematically.

---

## Part B: Coding

### Exercise B1 🟢 — Numerical Demonstration
```python
# TODO: Demonstrate numerical issues
# 1. Compute product of 1000 probabilities (each ~0.1)
# 2. Show underflow
# 3. Compute sum of log-probabilities
# 4. Compare results
```

### Exercise B2 🟡 — Log-Sum-Exp Implementation
```python
# TODO: Implement numerically stable log-sum-exp
# 1. Naive version
# 2. Stable version
# 3. Compare on extreme inputs
```

### Exercise B3 🔴 — MLE Without Log (Symbolic)
```python
# TODO: Using symbolic math (sympy)
# 1. Define Gaussian likelihood for 5 samples
# 2. Compute gradient of likelihood (product form)
# 3. Compute gradient of log-likelihood
# 4. Compare complexity
```

---

## Part C: Conceptual

### C1 🟡
Cross-entropy loss is $-\sum_i y_i \log(\hat{y}_i)$. Why is the log crucial for training neural networks?

### C2 🔴
Explain how the log function connects information theory, probability, and neural network training.
