# Tutorial 00: Probability Foundations — Solutions

---

## Part A: Theory & Derivation Solutions

### Solution A1 — Random Variable Basics

a) **Discrete** — the outcomes are countable (1, 2, 3, 4, 5, 6).

b) Sample space: $\Omega = \{1, 2, 3, 4, 5, 6\}$

c) PMF:
$$P(X = x) = \frac{1}{6} \text{ for } x \in \{1, 2, 3, 4, 5, 6\}$$

d) Verification:
$$\sum_{x=1}^{6} P(X = x) = 6 \times \frac{1}{6} = 1 \checkmark$$

---

### Solution A2 — PMF Properties

a) **Expected value**:
$$E[X] = \sum_x x \cdot P(X=x) = 1(0.2) + 2(0.3) + 3(0.5) = 0.2 + 0.6 + 1.5 = 2.3$$

b) **Second moment**:
$$E[X^2] = \sum_x x^2 \cdot P(X=x) = 1(0.2) + 4(0.3) + 9(0.5) = 0.2 + 1.2 + 4.5 = 5.9$$

c) **Variance**:
$$\text{Var}(X) = E[X^2] - (E[X])^2 = 5.9 - (2.3)^2 = 5.9 - 5.29 = 0.61$$

---

### Solution A3 — PDF Integration

a) **Finding c**:
$$\int_0^2 cx^2 dx = 1$$
$$c \cdot \frac{x^3}{3} \Big|_0^2 = 1$$
$$c \cdot \frac{8}{3} = 1$$
$$\boxed{c = \frac{3}{8}}$$

b) **Probability**:
$$P(0 \leq X \leq 1) = \int_0^1 \frac{3}{8}x^2 dx = \frac{3}{8} \cdot \frac{x^3}{3} \Big|_0^1 = \frac{3}{8} \cdot \frac{1}{3} = \boxed{\frac{1}{8}}$$

c) **CDF**:
$$F(x) = \int_0^x \frac{3}{8}t^2 dt = \frac{3}{8} \cdot \frac{t^3}{3} \Big|_0^x = \frac{x^3}{8}$$

$$F(x) = \begin{cases} 0 & x < 0 \\ \frac{x^3}{8} & 0 \leq x \leq 2 \\ 1 & x > 2 \end{cases}$$

d) **Expected value**:
$$E[X] = \int_0^2 x \cdot \frac{3}{8}x^2 dx = \frac{3}{8} \int_0^2 x^3 dx = \frac{3}{8} \cdot \frac{x^4}{4} \Big|_0^2 = \frac{3}{8} \cdot \frac{16}{4} = \frac{3}{8} \cdot 4 = \boxed{\frac{3}{2}}$$

---

### Solution A4 — The Continuous Paradox

a) $P(X = 0.5) = 0$

**Explanation**: For continuous distributions, the probability of any single point is zero because:
$$P(X = a) = \int_a^a f(x)dx = 0$$
An integral over a set of measure zero is zero. Intuitively, there are uncountably infinite points, and no single point can have positive probability while the total remains 1.

b) For uniform on $[0,1]$, $f(x) = 1$:
$$P(0.4 \leq X \leq 0.6) = \int_{0.4}^{0.6} 1 \, dx = 0.2$$

c) **Yes, PDF can exceed 1!**

Example: Uniform on $[0, 0.5]$ has $f(x) = 2$ for $x \in [0, 0.5]$.

This is fine because we integrate $f(x)$, not sum it:
$$\int_0^{0.5} 2 \, dx = 1 \checkmark$$

PDF is a **density**, not a probability!

---

### Solution A5 — Discrete to Continuous Transition

a) With $n = 4$ points at $\{0.125, 0.375, 0.625, 0.875\}$:
$$P(X = x_i) = \frac{1}{4} = 0.25$$

b) Entropy:
$$H = -\sum_{i=1}^{4} \frac{1}{4} \log \frac{1}{4} = -4 \cdot \frac{1}{4} \cdot (-\log 4) = \log 4 = 2 \text{ bits}$$

c) As $n \to \infty$:
$$H = \log n \to \infty$$

d) **Resolution of the paradox**:

Differential entropy differs from discrete entropy by a term $\log(\Delta x)$ where $\Delta x$ is the bin width. As $\Delta x \to 0$, $\log(\Delta x) \to -\infty$.

The discrete entropy is:
$$H_{discrete} = h_{differential} - \log(\Delta x)$$

For uniform on $[0,1]$: $h = 0$ bits, but discretizing with $n$ bins gives $H = \log n = -\log(1/n) = -\log(\Delta x)$.

This is why differential entropy can be negative (the "missing" infinite constant).

---

### Solution A6 — Joint Distribution Derivation

a) **Finding c**:
$$\int_0^1 \int_0^1 cxy \, dx \, dy = 1$$
$$c \int_0^1 y \left[\frac{x^2}{2}\right]_0^1 dy = 1$$
$$c \int_0^1 \frac{y}{2} dy = 1$$
$$c \cdot \frac{1}{2} \cdot \frac{1}{2} = 1$$
$$\boxed{c = 4}$$

b) **Marginals**:
$$f_X(x) = \int_0^1 4xy \, dy = 4x \cdot \frac{y^2}{2} \Big|_0^1 = 4x \cdot \frac{1}{2} = 2x$$

$$f_Y(y) = \int_0^1 4xy \, dx = 4y \cdot \frac{x^2}{2} \Big|_0^1 = 4y \cdot \frac{1}{2} = 2y$$

c) **Independence test**:
$$f(x,y) = 4xy = (2x)(2y) = f_X(x) \cdot f_Y(y) \checkmark$$

**Yes, they are independent!**

d) **Verify independence property**:
$$E[X] = \int_0^1 x \cdot 2x \, dx = 2 \int_0^1 x^2 dx = 2 \cdot \frac{1}{3} = \frac{2}{3}$$

By symmetry: $E[Y] = \frac{2}{3}$

$$E[XY] = \int_0^1 \int_0^1 xy \cdot 4xy \, dx \, dy = 4 \int_0^1 x^2 dx \int_0^1 y^2 dy = 4 \cdot \frac{1}{3} \cdot \frac{1}{3} = \frac{4}{9}$$

Check: $E[X] \cdot E[Y] = \frac{2}{3} \cdot \frac{2}{3} = \frac{4}{9} = E[XY] \checkmark$

---

### Solution A7 — Change of Variables

Let $Y = -\ln(X)$ where $X \sim \text{Uniform}(0, 1)$.

a) **CDF of Y**:
$$F_Y(y) = P(Y \leq y) = P(-\ln(X) \leq y) = P(\ln(X) \geq -y) = P(X \geq e^{-y})$$
$$= 1 - P(X < e^{-y}) = 1 - e^{-y} \quad \text{for } y \geq 0$$

b) **PDF of Y**:
$$f_Y(y) = \frac{d}{dy} F_Y(y) = \frac{d}{dy}(1 - e^{-y}) = e^{-y} \quad \text{for } y \geq 0$$

c) This is the **Exponential(1) distribution**! (rate $\lambda = 1$)

d) **Usefulness**: The inverse transform method lets us sample from ANY distribution:
   - Generate $U \sim \text{Uniform}(0, 1)$
   - Compute $X = F^{-1}(U)$
   - Then $X$ follows the distribution with CDF $F$

This works because $F(X)$ is always uniform when $X \sim F$.

---

## Part B: Coding Solutions

### Solution B1 — Simulate and Verify PMF

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate 10,000 die rolls
np.random.seed(42)
rolls = np.random.randint(1, 7, size=10000)

# Compute empirical frequencies
values, counts = np.unique(rolls, return_counts=True)
empirical_pmf = counts / len(rolls)

# Theoretical PMF
theoretical_pmf = np.ones(6) / 6

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(1, 7)
width = 0.35

ax.bar(x - width/2, empirical_pmf, width, label='Empirical', alpha=0.7)
ax.bar(x + width/2, theoretical_pmf, width, label='Theoretical', alpha=0.7)
ax.set_xlabel('Die Value')
ax.set_ylabel('Probability')
ax.set_title('Simulated vs Theoretical PMF of Fair Die')
ax.legend()
ax.set_xticks(x)

plt.tight_layout()
plt.show()

# Print comparison
print("Value | Empirical | Theoretical | Difference")
for i in range(6):
    print(f"  {i+1}   |  {empirical_pmf[i]:.4f}   |   {theoretical_pmf[i]:.4f}   |  {abs(empirical_pmf[i] - theoretical_pmf[i]):.4f}")
```

---

### Solution B2 — PDF Normalization Check

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Define unnormalized PDF
def f_unnorm(x, c=1):
    if 0 <= x <= 2:
        return c * x**2
    return 0

# Find normalization constant
integral, _ = integrate.quad(lambda x: x**2, 0, 2)
c = 1 / integral
print(f"Normalization constant c = {c:.6f}")
print(f"Expected: c = 3/8 = {3/8:.6f}")

# Verify integral = 1
def f_norm(x):
    return c * x**2 if 0 <= x <= 2 else 0

integral_check, _ = integrate.quad(f_norm, 0, 2)
print(f"Integral of normalized PDF: {integral_check:.6f}")

# Plot
x = np.linspace(-0.5, 2.5, 200)
y = [f_norm(xi) for xi in x]

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.fill_between(x, y, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'PDF: f(x) = {c:.4f}x² for 0 ≤ x ≤ 2')
plt.axhline(y=0, color='black', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Solution B3 — CDF from PDF

```python
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# Exponential distribution parameters
lam = 2.0

# PDF
def pdf(x):
    return lam * np.exp(-lam * x) if x >= 0 else 0

# Numerical CDF
def cdf_numerical(x_val):
    if x_val < 0:
        return 0
    result, _ = integrate.quad(pdf, 0, x_val)
    return result

# Analytical CDF
def cdf_analytical(x):
    return 1 - np.exp(-lam * x) if x >= 0 else 0

# Compare
x = np.linspace(0, 4, 100)
cdf_num = [cdf_numerical(xi) for xi in x]
cdf_ana = [cdf_analytical(xi) for xi in x]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PDF plot
pdf_vals = [pdf(xi) for xi in x]
axes[0].plot(x, pdf_vals, 'b-', linewidth=2)
axes[0].fill_between(x, pdf_vals, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title(f'Exponential PDF (λ = {lam})')

# CDF comparison
axes[1].plot(x, cdf_num, 'b-', linewidth=2, label='Numerical')
axes[1].plot(x, cdf_ana, 'r--', linewidth=2, label='Analytical')
axes[1].set_xlabel('x')
axes[1].set_ylabel('F(x)')
axes[1].set_title('CDF: Numerical vs Analytical')
axes[1].legend()

plt.tight_layout()
plt.show()

# Verify they match
max_diff = max(abs(n - a) for n, a in zip(cdf_num, cdf_ana))
print(f"Maximum difference between numerical and analytical CDF: {max_diff:.2e}")
```

---

### Solution B4 — Visualize Joint Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

# Joint distribution of two fair dice
joint = np.ones((6, 6)) / 36

# Visualize joint distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Joint as heatmap
im = axes[0, 0].imshow(joint, cmap='Blues')
axes[0, 0].set_xlabel('Die 2')
axes[0, 0].set_ylabel('Die 1')
axes[0, 0].set_title('Joint Distribution P(X, Y)')
axes[0, 0].set_xticks(range(6))
axes[0, 0].set_xticklabels(range(1, 7))
axes[0, 0].set_yticks(range(6))
axes[0, 0].set_yticklabels(range(1, 7))
plt.colorbar(im, ax=axes[0, 0])

# Marginals
marginal_x = joint.sum(axis=1)
marginal_y = joint.sum(axis=0)

axes[0, 1].bar(range(1, 7), marginal_x, color='steelblue', alpha=0.7)
axes[0, 1].set_xlabel('Die 1 Value')
axes[0, 1].set_ylabel('Probability')
axes[0, 1].set_title('Marginal P(X)')

axes[1, 0].bar(range(1, 7), marginal_y, color='coral', alpha=0.7)
axes[1, 0].set_xlabel('Die 2 Value')
axes[1, 0].set_ylabel('Probability')
axes[1, 0].set_title('Marginal P(Y)')

# Distribution of sum
sum_probs = {}
for i in range(6):
    for j in range(6):
        s = (i + 1) + (j + 1)
        sum_probs[s] = sum_probs.get(s, 0) + joint[i, j]

sums = sorted(sum_probs.keys())
probs = [sum_probs[s] for s in sums]

axes[1, 1].bar(sums, probs, color='green', alpha=0.7)
axes[1, 1].set_xlabel('Sum X + Y')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_title('Distribution of Sum')

plt.tight_layout()
plt.show()

print("Sum distribution:")
for s, p in zip(sums, probs):
    print(f"  P(X+Y = {s:2d}) = {p:.4f} = {int(p*36)}/36")
```

---

### Solution B5 — Inverse Transform Sampling

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Parameters
lam = 2.0
n_samples = 10000

# Method 1: Inverse transform sampling
U = np.random.uniform(0, 1, n_samples)
X_inverse = -np.log(U) / lam  # Inverse of CDF: F^{-1}(u) = -ln(1-u)/λ ≈ -ln(u)/λ

# Method 2: Direct sampling (for comparison)
X_direct = np.random.exponential(1/lam, n_samples)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(0, 4, 100)
pdf_true = lam * np.exp(-lam * x)

axes[0].hist(X_inverse, bins=50, density=True, alpha=0.7, label='Inverse Transform')
axes[0].plot(x, pdf_true, 'r-', linewidth=2, label='True PDF')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].set_title('Inverse Transform Sampling')
axes[0].legend()

axes[1].hist(X_direct, bins=50, density=True, alpha=0.7, label='numpy.random.exponential')
axes[1].plot(x, pdf_true, 'r-', linewidth=2, label='True PDF')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Density')
axes[1].set_title('Direct Sampling (comparison)')
axes[1].legend()

plt.tight_layout()
plt.show()

# Generalized inverse transform sampler
def inverse_transform_sample(cdf_inverse, n):
    """Sample using inverse transform method."""
    U = np.random.uniform(0, 1, n)
    return cdf_inverse(U)

# Example: Sample from Weibull distribution
# CDF: F(x) = 1 - exp(-(x/λ)^k)
# CDF inverse: F^{-1}(u) = λ * (-ln(1-u))^(1/k)
k, scale = 2, 1
weibull_inverse = lambda u: scale * (-np.log(1 - u))**(1/k)

X_weibull = inverse_transform_sample(weibull_inverse, 10000)

plt.figure(figsize=(10, 5))
plt.hist(X_weibull, bins=50, density=True, alpha=0.7, label='Sampled')
x = np.linspace(0, 4, 100)
plt.plot(x, stats.weibull_min.pdf(x, k, scale=scale), 'r-', linewidth=2, label='True Weibull PDF')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'Weibull(k={k}, λ={scale}) via Inverse Transform')
plt.legend()
plt.show()
```

---

### Solution B6 — Kernel Density Estimation

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate mixture of two Gaussians
n_samples = 1000
mix_weight = 0.3
mu1, sigma1 = -2, 0.5
mu2, sigma2 = 2, 1.0

# Generate samples
n1 = int(n_samples * mix_weight)
n2 = n_samples - n1
samples = np.concatenate([
    np.random.normal(mu1, sigma1, n1),
    np.random.normal(mu2, sigma2, n2)
])

# True PDF
x = np.linspace(-5, 6, 200)
true_pdf = (mix_weight * stats.norm.pdf(x, mu1, sigma1) + 
            (1 - mix_weight) * stats.norm.pdf(x, mu2, sigma2))

# KDE with different bandwidths
bandwidths = [0.1, 0.3, 0.5, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, bw in zip(axes, bandwidths):
    kde = stats.gaussian_kde(samples, bw_method=bw)
    estimated_pdf = kde(x)
    
    ax.hist(samples, bins=50, density=True, alpha=0.3, color='gray', label='Histogram')
    ax.plot(x, true_pdf, 'b-', linewidth=2, label='True PDF')
    ax.plot(x, estimated_pdf, 'r--', linewidth=2, label='KDE estimate')
    ax.set_xlabel('x')
    ax.set_ylabel('Density')
    ax.set_title(f'Bandwidth = {bw}')
    ax.legend()

plt.tight_layout()
plt.show()

# Scott's rule (default) vs Silverman's rule
kde_scott = stats.gaussian_kde(samples, bw_method='scott')
kde_silverman = stats.gaussian_kde(samples, bw_method='silverman')

plt.figure(figsize=(10, 6))
plt.plot(x, true_pdf, 'b-', linewidth=2, label='True PDF')
plt.plot(x, kde_scott(x), 'r--', linewidth=2, label=f"Scott's rule (bw={kde_scott.factor:.3f})")
plt.plot(x, kde_silverman(x), 'g:', linewidth=2, label=f"Silverman's rule (bw={kde_silverman.factor:.3f})")
plt.xlabel('x')
plt.ylabel('Density')
plt.title('KDE with Automatic Bandwidth Selection')
plt.legend()
plt.show()
```

---

## Part C: Conceptual Solutions

### Solution C1
**PMF values represent actual probabilities**, which by definition must be ≤ 1.

**PDF values represent density** — probability per unit length. A PDF can exceed 1 as long as the integral (total area) equals 1.

Example: Uniform on [0, 0.1] has PDF = 10, but ∫₀^{0.1} 10 dx = 1.

---

### Solution C2
For **discrete**: $P(A) = 0$ means $A$ is impossible (never happens).

For **continuous**: $P(A) = 0$ does NOT mean impossible!
- Single points have probability 0 but can still occur
- "Almost surely" vs "surely" distinction in measure theory
- Example: Pick a random real in [0,1]. P(X = 0.5) = 0, but 0.5 is a valid outcome.

---

### Solution C3
The fundamental difference is **countable vs uncountable**:
- **Discrete**: Finite or countably infinite outcomes → can sum individual probabilities
- **Continuous**: Uncountably infinite outcomes → individual points have probability 0, must integrate density

The PMF assigns mass to points; the PDF describes how mass is spread continuously.

---

### Solution C4
**Resolution**: Discrete and differential entropy measure different things.

Discrete entropy H measures absolute uncertainty (in bits).

Differential entropy h measures uncertainty relative to a reference (uniform density).

The relationship:
$$H_{discrete} = h_{differential} - \log(\Delta x)$$

As discretization $\Delta x \to 0$, the $-\log(\Delta x) \to +\infty$ term diverges.

This is why:
- Differential entropy can be negative (less uncertain than reference)
- Comparing differential entropies is meaningful (differences cancel the infinite constant)
- KL divergence is well-defined for continuous distributions (the constant cancels!)
