# Tutorial 03: Normal Distributions — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Standard Normal Properties
For $Z \sim N(0, 1)$:
a) What is $E[Z]$?
b) What is $E[Z^2]$?
c) What is $E[Z^3]$?
d) Prove your answer for (c).

### Exercise A2 🟢 — Linear Transformation
If $X \sim N(\mu, \sigma^2)$, show that $Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$.

### Exercise A3 🟡 — Normalization Constant
Prove that $\int_{-\infty}^{\infty} e^{-x^2/2} dx = \sqrt{2\pi}$ using polar coordinates.

### Exercise A4 🟡 — Maximum Entropy Derivation
Prove that the Gaussian is the maximum entropy distribution for a given mean $\mu$ and variance $\sigma^2$.

### Exercise A5 🔴 — Multivariate Gaussian Derivation
For $\mathbf{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:
a) Write the PDF formula
b) Derive the marginal distribution of a single component $X_i$
c) Show that the conditional $p(X_1 | X_2 = x_2)$ is also Gaussian

### Exercise A6 🔴 — Sum of Gaussians
Prove: If $X \sim N(\mu_1, \sigma_1^2)$ and $Y \sim N(\mu_2, \sigma_2^2)$ are independent, then:
$$X + Y \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

---

## Part B: Coding

### Exercise B1 🟢 — Sampling and Visualization
```python
# TODO: Generate samples from N(μ, σ²) and plot:
# 1. Histogram with theoretical PDF overlay
# 2. Q-Q plot to verify normality
# 3. Verify empirical mean and variance match parameters
```

### Exercise B2 🟡 — Central Limit Theorem Demo
```python
# TODO: Demonstrate CLT
# 1. Generate samples from Uniform(0,1)
# 2. Compute mean of n samples, repeat many times
# 3. Show distribution approaches Gaussian as n increases
# 4. Plot for n = 1, 2, 5, 30
```

### Exercise B3 🟡 — Multivariate Gaussian Visualization
```python
# TODO: Create 2D Gaussian visualization
# 1. Define mean vector and covariance matrix
# 2. Plot contours of constant probability
# 3. Show how correlation affects the ellipse orientation
# 4. Generate and plot samples
```

### Exercise B4 🔴 — MLE for Gaussian Parameters
```python
# TODO: Maximum Likelihood Estimation
# 1. Generate samples from known Gaussian
# 2. Derive and implement MLE formulas for μ and σ²
# 3. Show consistency: estimates converge as n → ∞
# 4. Compare biased vs unbiased variance estimators
```

---

## Part C: Conceptual

### C1 🟡
Why is the Gaussian distribution so common in nature? (Hint: CLT and entropy)

### C2 🔴
The covariance matrix $\Sigma$ must be positive semi-definite. Why? What would happen geometrically if it weren't?
