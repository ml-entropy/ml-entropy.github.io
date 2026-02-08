# Tutorial 3: Normal & Multivariate Normal Distributions

## 🎯 The Big Picture

**Gaussians are everywhere in ML because they maximize entropy given constraints.**

If all you know is the mean and variance of a distribution, the **maximum entropy** choice is Gaussian. Nature tends toward maximum entropy, and we often only know first and second moments—hence Gaussians appear everywhere.

---

## 1. Why Gaussians? The Maximum Entropy Principle

### The Principle

Among all distributions with a given mean $\mu$ and variance $\sigma^2$, the **Gaussian** has maximum differential entropy:

$$h(X) = -\int p(x) \log p(x) \, dx$$

For a Gaussian: $h(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2} \log(2\pi e \sigma^2)$

### Intuition

Maximum entropy = maximum uncertainty = least assumptions.

If you only know the mean and variance:
- Assuming Gaussian adds **no extra information**
- Any other distribution would impose additional (unjustified) structure
- Gaussian is the "most honest" choice

### Connection to Central Limit Theorem

The CLT says: sums of many independent random variables → Gaussian.

Information-theoretic view: Adding independent random variables **increases entropy**, and the limit of maximum entropy given variance is Gaussian.

---

## 2. The Univariate Normal Distribution

### Definition

$$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Parameters:**
- $\mu$ = mean (location)
- $\sigma^2$ = variance (spread)
- $\sigma$ = standard deviation

### Key Properties

| Property | Value |
|----------|-------|
| Mean | $\mathbb{E}[X] = \mu$ |
| Variance | $\text{Var}[X] = \sigma^2$ |
| Mode | $\mu$ |
| Entropy | $\frac{1}{2}\log(2\pi e\sigma^2)$ |
| Support | $(-\infty, +\infty)$ |

### The 68-95-99.7 Rule

- 68% of probability within $\mu \pm \sigma$
- 95% within $\mu \pm 2\sigma$
- 99.7% within $\mu \pm 3\sigma$

### Log Probability

$$\log \mathcal{N}(x | \mu, \sigma^2) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

The squared term is why:
- **MSE loss** corresponds to Gaussian likelihood
- **Gaussian noise assumption** leads to L2 loss

---

## 3. The Multivariate Normal Distribution

### Definition

For $\mathbf{x} \in \mathbb{R}^d$:

$$\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**Parameters:**
- $\boldsymbol{\mu} \in \mathbb{R}^d$ = mean vector
- $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ = covariance matrix (symmetric, positive definite)

### The Covariance Matrix

$$\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T]$$

$$\boldsymbol{\Sigma}_{ij} = \text{Cov}(x_i, x_j) = \mathbb{E}[(x_i - \mu_i)(x_j - \mu_j)]$$

**Diagonal entries:** Variances $\sigma_i^2$
**Off-diagonal entries:** Covariances $\sigma_{ij}$

### Correlation Matrix

$$\boldsymbol{\rho}_{ij} = \frac{\boldsymbol{\Sigma}_{ij}}{\sigma_i \sigma_j}$$

Normalized covariance: $\rho_{ij} \in [-1, 1]$

### Geometric Interpretation

The covariance matrix encodes:
1. **Eigenvalues** → lengths of principal axes
2. **Eigenvectors** → directions of principal axes
3. **Determinant** → "volume" of the ellipsoid

Contours of equal probability are **ellipsoids** centered at $\boldsymbol{\mu}$.

---

## 4. The Mahalanobis Distance

### Definition

$$d_M(\mathbf{x}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

### Intuition

The Mahalanobis distance:
- Measures distance **in units of standard deviation**
- Accounts for **correlations** between dimensions
- Is **coordinate-independent** (rotation invariant)

For the standard normal ($\boldsymbol{\Sigma} = \mathbf{I}$), Mahalanobis = Euclidean distance.

### Why It Matters

The Gaussian pdf depends only on Mahalanobis distance:

$$\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) \propto \exp\left(-\frac{1}{2} d_M^2(\mathbf{x})\right)$$

Points with equal Mahalanobis distance have equal probability density.

---

## 5. Special Cases of Covariance

### Spherical Covariance

$$\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$$

- All dimensions have same variance
- No correlation between dimensions
- Contours are **circles/spheres**

### Diagonal Covariance

$$\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$$

- Different variance per dimension
- No correlation between dimensions
- Contours are **axis-aligned ellipses**
- **Used in VAEs** for efficiency

### Full Covariance

$$\boldsymbol{\Sigma} = \text{any positive definite matrix}$$

- Different variances
- Arbitrary correlations
- Contours are **rotated ellipses**

### Parameter Count

| Type | Parameters |
|------|-----------|
| Spherical | 1 |
| Diagonal | $d$ |
| Full | $d + \frac{d(d+1)}{2}$ |

---

## 6. Entropy of Multivariate Gaussians

### Formula

$$h(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})) = \frac{d}{2}(1 + \log 2\pi) + \frac{1}{2}\log|\boldsymbol{\Sigma}|$$

### Interpretation

Entropy depends only on:
- Dimensionality $d$
- Determinant of covariance $|\boldsymbol{\Sigma}|$ (the "volume")

**Not on the mean!** Location doesn't affect uncertainty.

### Maximum Entropy Property

Among all $d$-dimensional distributions with covariance $\boldsymbol{\Sigma}$:

$$h(X) \leq \frac{d}{2}(1 + \log 2\pi) + \frac{1}{2}\log|\boldsymbol{\Sigma}|$$

with equality **only for Gaussians**.

---

## 7. KL Divergence Between Multivariate Gaussians

### Formula

For $P = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ and $Q = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$:

$$D_{KL}(P \| Q) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} - d + \text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]$$

### Components

1. $\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}$: Ratio of "volumes"
2. $\text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1)$: How covariances align
3. $(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)$: Squared Mahalanobis distance between means

### VAE Special Case

For $P = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $Q = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$D_{KL}(P \| Q) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right)$$

This simplification is **crucial** for efficient VAE training.

---

## 8. Sampling from Multivariate Gaussians

### The Reparameterization Trick

To sample $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$:

1. Sample $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
2. Compute $\mathbf{x} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$

where $\mathbf{L}$ is the Cholesky decomposition: $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$

### For Diagonal Covariance (VAE)

$$\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$

where $\odot$ is element-wise multiplication.

**Why this matters:** Gradients flow through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$, enabling backpropagation!

---

## 9. Gaussians in Deep Learning

### Why Use Gaussians?

1. **Maximum entropy**: Minimal assumptions
2. **Closed-form KL**: Efficient computation
3. **Reparameterization**: Differentiable sampling
4. **Conjugacy**: Nice mathematical properties
5. **CLT**: Sums of effects → approximately Gaussian

### Common Uses

| Application | Gaussian Used For |
|-------------|------------------|
| VAE encoder | Posterior $q(z|x)$ |
| VAE decoder | Likelihood $p(x|z)$ |
| Gaussian processes | Function distributions |
| Bayesian neural networks | Weight distributions |
| Normalizing flows | Base distribution |
| Diffusion models | Noise schedule |

### The Gaussian Assumption in VAEs

VAEs assume:
- **Posterior** $q(z|x) = \mathcal{N}(\boldsymbol{\mu}_\phi(x), \text{diag}(\boldsymbol{\sigma}^2_\phi(x)))$
- **Prior** $p(z) = \mathcal{N}(\mathbf{0}, \mathbf{I})$

This gives us:
- Closed-form KL divergence
- Efficient sampling via reparameterization
- Smooth, continuous latent space

---

## 10. Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Univariate PDF | $\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$ |
| Multivariate PDF | $\frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}d_M^2\right)$ |
| Mahalanobis | $\sqrt{(\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ |
| Entropy | $\frac{d}{2}(1 + \log 2\pi) + \frac{1}{2}\log|\boldsymbol{\Sigma}|$ |
| Reparam trick | $\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ |

### Key Insights

1. **Gaussians maximize entropy** given mean and variance
2. **Covariance matrix encodes geometry** of the distribution
3. **Mahalanobis distance** is the natural distance metric
4. **Diagonal covariance** simplifies computation (VAEs)
5. **Reparameterization** enables gradient-based learning

---

## Next: VAE, ELBO, and Variational Inference

Now we have all the pieces to understand VAEs:
- Entropy and information theory ✓
- KL divergence ✓
- Gaussian distributions ✓

Let's put them together!
