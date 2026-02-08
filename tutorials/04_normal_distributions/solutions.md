# Tutorial 03: Normal Distributions — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Standard Normal Properties

For $Z \sim N(0, 1)$:

a) $E[Z] = 0$ (by symmetry)

b) $E[Z^2] = Var(Z) + (E[Z])^2 = 1 + 0 = 1$

c) $E[Z^3] = 0$

d) **Proof**: The PDF $f(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ is even (symmetric about 0).
   Since $z^3$ is odd and $f(z)$ is even, $z^3 f(z)$ is odd.
   The integral of an odd function over a symmetric interval is zero:
   $$E[Z^3] = \int_{-\infty}^{\infty} z^3 f(z) dz = 0$$

---

### Solution A2 — Linear Transformation

If $X \sim N(\mu, \sigma^2)$, then $X = \mu + \sigma Z$ where $Z \sim N(0,1)$.

Therefore:
$$Z = \frac{X - \mu}{\sigma}$$

Since linear transformations of Gaussians are Gaussian:
- $E[Z] = \frac{E[X] - \mu}{\sigma} = \frac{\mu - \mu}{\sigma} = 0$
- $Var(Z) = \frac{Var(X)}{\sigma^2} = \frac{\sigma^2}{\sigma^2} = 1$

Thus $Z \sim N(0, 1)$. ∎

---

### Solution A3 — Normalization Constant (Gaussian Integral)

Let $I = \int_{-\infty}^{\infty} e^{-x^2/2} dx$

**Trick**: Compute $I^2$ using polar coordinates:
$$I^2 = \int_{-\infty}^{\infty} e^{-x^2/2} dx \cdot \int_{-\infty}^{\infty} e^{-y^2/2} dy = \int\int e^{-(x^2+y^2)/2} dx\, dy$$

Convert to polar: $x^2 + y^2 = r^2$, $dx\,dy = r\,dr\,d\theta$:
$$I^2 = \int_0^{2\pi} \int_0^{\infty} e^{-r^2/2} r\, dr\, d\theta$$

Let $u = r^2/2$, $du = r\,dr$:
$$I^2 = 2\pi \int_0^{\infty} e^{-u} du = 2\pi \cdot 1 = 2\pi$$

Therefore: $\boxed{I = \sqrt{2\pi}}$ ∎

---

### Solution A4 — Maximum Entropy Derivation

We want to maximize $H[p] = -\int p(x) \log p(x) dx$ subject to:
1. $\int p(x) dx = 1$
2. $\int x \cdot p(x) dx = \mu$
3. $\int (x-\mu)^2 p(x) dx = \sigma^2$

Using Lagrange multipliers:
$$\mathcal{L} = -\int p \log p\, dx - \lambda_0 \left(\int p\, dx - 1\right) - \lambda_1 \left(\int xp\, dx - \mu\right) - \lambda_2 \left(\int (x-\mu)^2 p\, dx - \sigma^2\right)$$

Taking functional derivative $\frac{\delta \mathcal{L}}{\delta p} = 0$:
$$-\log p(x) - 1 - \lambda_0 - \lambda_1 x - \lambda_2 (x-\mu)^2 = 0$$

Solving:
$$p(x) = \exp\left(-1 - \lambda_0 - \lambda_1 x - \lambda_2(x-\mu)^2\right)$$

This is a Gaussian! Applying constraints gives $\lambda_1 = 0$, $\lambda_2 = \frac{1}{2\sigma^2}$.

$$\boxed{p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/2\sigma^2}}$$

---

### Solution A5 — Multivariate Gaussian

a) PDF:
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

b) Marginal of $X_i$: $X_i \sim N(\mu_i, \Sigma_{ii})$

Partition: $\mathbf{X} = \begin{pmatrix} X_i \\ \mathbf{X}_{-i} \end{pmatrix}$

The marginal is obtained by integrating out $\mathbf{X}_{-i}$, which preserves the Gaussian form with parameters from the corresponding diagonal entry.

c) Conditional: $X_1 | X_2 = x_2$ is Gaussian with:
$$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)$$
$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$

---

### Solution A6 — Sum of Gaussians

Use characteristic functions (Fourier transforms).

For $X \sim N(\mu, \sigma^2)$: $\phi_X(t) = e^{i\mu t - \sigma^2 t^2/2}$

For independent RVs, $\phi_{X+Y}(t) = \phi_X(t) \cdot \phi_Y(t)$:
$$\phi_{X+Y}(t) = e^{i\mu_1 t - \sigma_1^2 t^2/2} \cdot e^{i\mu_2 t - \sigma_2^2 t^2/2} = e^{i(\mu_1+\mu_2)t - (\sigma_1^2+\sigma_2^2)t^2/2}$$

This is the characteristic function of $N(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$. ∎

---

## Part B: Coding Solutions

### Solution B1 — Sampling and Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Parameters
mu, sigma = 5, 2
n_samples = 10000

# Generate samples
samples = np.random.normal(mu, sigma, n_samples)

# Verify statistics
print(f"True: μ={mu}, σ={sigma}")
print(f"Sample: μ̂={np.mean(samples):.3f}, σ̂={np.std(samples):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram with PDF overlay
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
axes[0].hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='True PDF')
axes[0].set_xlabel('x')
axes[0].set_title(f'N({mu}, {sigma}²) - Histogram')
axes[0].legend()

# Q-Q plot
stats.probplot(samples, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()
```

### Solution B2 — Central Limit Theorem Demo

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n_experiments = 10000
ns = [1, 2, 5, 30]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, n in zip(axes.flatten(), ns):
    # Generate uniform samples and compute means
    samples = np.random.uniform(0, 1, size=(n_experiments, n))
    means = samples.mean(axis=1)
    
    # Standardize
    z = (means - 0.5) / (1/np.sqrt(12*n))
    
    # Plot
    ax.hist(z, bins=50, density=True, alpha=0.7, label='Sample means')
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    ax.set_title(f'n = {n}')
    ax.set_xlim(-4, 4)
    ax.legend()

plt.suptitle('Central Limit Theorem: Mean of Uniform RVs → Gaussian')
plt.tight_layout()
plt.show()
```

### Solution B3 — Multivariate Gaussian

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Define parameters
mu = np.array([0, 0])
correlations = [0, 0.5, 0.9, -0.7]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, rho in zip(axes.flatten(), correlations):
    # Covariance matrix
    Sigma = np.array([[1, rho], [rho, 1]])
    
    # Create grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    # Compute PDF
    rv = stats.multivariate_normal(mu, Sigma)
    Z = rv.pdf(pos)
    
    # Plot contours
    ax.contour(X, Y, Z, levels=5)
    
    # Generate and plot samples
    samples = rv.rvs(200)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)
    
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_title(f'ρ = {rho}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('2D Gaussian: Effect of Correlation')
plt.tight_layout()
plt.show()
```

### Solution B4 — MLE for Gaussian Parameters

```python
import numpy as np
import matplotlib.pyplot as plt

# True parameters
mu_true, sigma_true = 3, 2

# MLE formulas (derived from maximizing log-likelihood):
# μ̂_MLE = (1/n) Σ xᵢ
# σ²̂_MLE = (1/n) Σ (xᵢ - μ̂)²  (biased)
# σ²̂_unbiased = (1/(n-1)) Σ (xᵢ - μ̂)²

np.random.seed(42)
sample_sizes = np.logspace(1, 4, 50, dtype=int)
n_trials = 100

mu_estimates = []
sigma_biased = []
sigma_unbiased = []

for n in sample_sizes:
    mu_est = []
    s_biased = []
    s_unbiased = []
    
    for _ in range(n_trials):
        samples = np.random.normal(mu_true, sigma_true, n)
        
        # MLE estimates
        mu_hat = np.mean(samples)
        var_biased = np.var(samples, ddof=0)  # MLE (biased)
        var_unbiased = np.var(samples, ddof=1)  # Unbiased
        
        mu_est.append(mu_hat)
        s_biased.append(np.sqrt(var_biased))
        s_unbiased.append(np.sqrt(var_unbiased))
    
    mu_estimates.append(np.mean(mu_est))
    sigma_biased.append(np.mean(s_biased))
    sigma_unbiased.append(np.mean(s_unbiased))

# Plot convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].axhline(mu_true, color='r', linestyle='--', label='True μ')
axes[0].plot(sample_sizes, mu_estimates, 'b-', label='MLE μ̂')
axes[0].set_xscale('log')
axes[0].set_xlabel('Sample size n')
axes[0].set_ylabel('μ estimate')
axes[0].legend()
axes[0].set_title('MLE for Mean (consistent)')

axes[1].axhline(sigma_true, color='r', linestyle='--', label='True σ')
axes[1].plot(sample_sizes, sigma_biased, 'b-', label='Biased (MLE)')
axes[1].plot(sample_sizes, sigma_unbiased, 'g-', label='Unbiased')
axes[1].set_xscale('log')
axes[1].set_xlabel('Sample size n')
axes[1].set_ylabel('σ estimate')
axes[1].legend()
axes[1].set_title('MLE for Std Dev (bias vanishes as n→∞)')

plt.tight_layout()
plt.show()

print(f"Note: MLE variance σ²̂ = (n-1)/n × unbiased estimator")
print(f"Bias → 0 as n → ∞ (consistent)")
```

---

## Part C: Conceptual Solutions

### C1
The Gaussian is ubiquitous because:
1. **Central Limit Theorem**: Sum of many independent effects → Gaussian
2. **Maximum Entropy**: Among all distributions with fixed mean and variance, Gaussian has maximum entropy (most "random")
3. **Stability**: Sums of Gaussians are Gaussian (closed under addition)

### C2
$\Sigma$ must be positive semi-definite because:
1. **Variance is non-negative**: $Var(\mathbf{a}^T\mathbf{X}) = \mathbf{a}^T\Sigma\mathbf{a} \geq 0$ for all $\mathbf{a}$
2. **Geometric interpretation**: The PDF involves $\exp(-\frac{1}{2}\mathbf{x}^T\Sigma^{-1}\mathbf{x})$. If $\Sigma$ had negative eigenvalues, this would explode instead of decay, making it impossible to normalize.
3. **Eigenvalues = variances** along principal axes. Negative variance is meaningless.
