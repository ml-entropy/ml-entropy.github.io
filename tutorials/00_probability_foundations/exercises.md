# Tutorial 00: Probability Foundations — Exercises

## Difficulty Levels
- 🟢 Easy: Direct application of concepts
- 🟡 Medium: Requires combining concepts
- 🔴 Hard: Requires deeper reasoning or derivation

---

## Part A: Theory & Derivation Exercises

### Exercise A1 🟢 — Random Variable Basics
A random variable $X$ represents the outcome of rolling a fair 6-sided die.

a) Is $X$ discrete or continuous?
b) What is the sample space?
c) Write out the PMF $P(X = x)$ for all possible $x$.
d) Verify that your PMF sums to 1.

---

### Exercise A2 🟢 — PMF Properties
Given a PMF: $P(X = 1) = 0.2$, $P(X = 2) = 0.3$, $P(X = 3) = 0.5$

a) Compute $E[X]$ (expected value).
b) Compute $E[X^2]$.
c) Compute $\text{Var}(X) = E[X^2] - (E[X])^2$.

---

### Exercise A3 🟡 — PDF Integration
A continuous random variable has PDF:
$$f(x) = \begin{cases} cx^2 & 0 \leq x \leq 2 \\ 0 & \text{otherwise} \end{cases}$$

a) Find the constant $c$ that makes this a valid PDF.
b) Compute $P(0 \leq X \leq 1)$.
c) Compute the CDF $F(x)$.
d) Compute $E[X]$.

---

### Exercise A4 🟡 — The Continuous Paradox
For a continuous uniform distribution on $[0, 1]$:

a) What is $P(X = 0.5)$? Explain why.
b) What is $P(0.4 \leq X \leq 0.6)$?
c) Can $f(x) > 1$ for some PDF? Give an example.

---

### Exercise A5 🟡 — Discrete to Continuous Transition
Consider approximating a continuous uniform distribution on $[0, 1]$ with a discrete distribution.

a) With $n = 4$ equally spaced points, what are the probabilities?
b) Write the entropy of this discrete approximation.
c) As $n \to \infty$, what happens to the entropy? (Hint: it diverges)
d) Why does this explain the "constant" in differential entropy?

---

### Exercise A6 🔴 — Joint Distribution Derivation
Two random variables $X$ and $Y$ have joint PDF:
$$f(x, y) = \begin{cases} cxy & 0 \leq x \leq 1, 0 \leq y \leq 1 \\ 0 & \text{otherwise} \end{cases}$$

a) Find $c$.
b) Find the marginal PDFs $f_X(x)$ and $f_Y(y)$.
c) Are $X$ and $Y$ independent? Prove it.
d) Compute $E[XY]$ and verify $E[XY] = E[X] \cdot E[Y]$ (for independent variables).

---

### Exercise A7 🔴 — Change of Variables
If $X$ is uniform on $[0, 1]$, and $Y = -\ln(X)$:

a) What is the CDF of $Y$?
b) What is the PDF of $Y$?
c) What distribution is this? (Hint: exponential)
d) This is called the "inverse transform method" — why is it useful?

---

## Part B: Coding Exercises

### Exercise B1 🟢 — Simulate and Verify PMF
```python
# TODO: Simulate rolling a fair die 10,000 times
# Compare empirical frequencies to theoretical PMF
# Plot both as a bar chart
```

---

### Exercise B2 🟢 — PDF Normalization Check
```python
# TODO: Given f(x) = c * x^2 for x in [0, 2]
# 1. Use scipy.integrate.quad to find c numerically
# 2. Verify ∫f(x)dx = 1
# 3. Plot the PDF
```

---

### Exercise B3 🟡 — CDF from PDF
```python
# TODO: For the exponential distribution f(x) = λ*exp(-λx) for x ≥ 0
# 1. Implement the PDF
# 2. Numerically compute the CDF using cumulative integration
# 3. Compare to the analytical CDF: F(x) = 1 - exp(-λx)
# 4. Plot both
```

---

### Exercise B4 🟡 — Visualize Joint Distribution
```python
# TODO: Create a joint distribution for two dice rolls
# 1. Create a 6x6 matrix of joint probabilities
# 2. Visualize as a heatmap
# 3. Compute and plot marginal distributions
# 4. Compute the distribution of the sum X + Y
```

---

### Exercise B5 🔴 — Inverse Transform Sampling
```python
# TODO: Implement inverse transform sampling
# 1. Generate uniform samples U ~ Uniform(0, 1)
# 2. Transform to exponential: X = -ln(U)/λ
# 3. Verify by comparing histogram to exponential PDF
# 4. Generalize: sample from any distribution given its CDF inverse
```

---

### Exercise B6 🔴 — Kernel Density Estimation
```python
# TODO: Approximate a PDF from samples
# 1. Generate 1000 samples from a mixture of two Gaussians
# 2. Use scipy.stats.gaussian_kde to estimate the PDF
# 3. Compare estimated PDF to true PDF
# 4. Experiment with different bandwidth parameters
```

---

## Part C: Conceptual Questions

### Exercise C1 🟢
Why can't a PMF value ever exceed 1, but a PDF value can?

### Exercise C2 🟡
If $P(A) = 0$, does that mean event $A$ is impossible? Discuss for discrete vs continuous cases.

### Exercise C3 🟡
Explain in your own words why $\sum_x p(x) = 1$ for PMF but $\int f(x) dx = 1$ for PDF. What's the fundamental difference?

### Exercise C4 🔴
The entropy of a discrete uniform distribution over $n$ outcomes is $\log n$. As $n \to \infty$, this goes to $\infty$. But continuous distributions have finite differential entropy. Resolve this apparent paradox.
