# Tutorial 17: Normalizing Flows

## 🎯 Why This Tutorial?

**Prerequisites:** Probability distributions (Tutorial 00), KL divergence (Tutorial 03), neural networks (Tutorial 07), GANs (Tutorial 16)

Normalizing flows offer something remarkable: the ability to compute **exact likelihoods** while generating high-quality samples. Unlike GANs (no likelihood) or VAEs (approximate likelihood), flows give you the best of both worlds — if you're willing to accept architectural constraints.

The core idea is breathtakingly elegant: start with a simple distribution (Gaussian), apply a series of **invertible transformations**, and track exactly how the probability density changes at each step using the **Jacobian determinant**.

After this tutorial, you'll understand:
- How the change of variables formula works in 1D and higher dimensions
- Why the Jacobian measures how transformations stretch and squeeze space
- How to compose simple transformations into complex distributions
- Why coupling layers make flows computationally tractable
- When to choose flows over GANs, VAEs, or diffusion models

---

## 1. The Density Estimation Problem

Given data $\{x_1, ..., x_N\}$, we want to find a probability density $p_\theta(x)$ that:
1. Assigns high probability to observed data (good fit)
2. Can generate new samples (generative capability)
3. Allows evaluating $p_\theta(x)$ for any $x$ (density evaluation)

Normalizing flows achieve ALL THREE — unique among deep generative models.

---

## 2. Change of Variables: The Foundation

### 2.1 One Dimension

If $x$ has density $p_X(x)$ and $y = f(x)$ is a monotonic differentiable function, then:

$$p_Y(y) = p_X(f^{-1}(y)) \cdot \left|\frac{df^{-1}}{dy}\right|$$

**Why the absolute derivative?** When $f$ stretches a region of space, the same probability must spread over a larger region, so density **decreases**. When $f$ compresses a region, density **increases**.

**Example:** Let $X \sim \mathcal{N}(0, 1)$ and $Y = 2X + 1$. Then:
- $f^{-1}(y) = (y-1)/2$
- $|df^{-1}/dy| = 1/2$
- $p_Y(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(y-1)^2}{8}\right) \cdot \frac{1}{2}$

This is $\mathcal{N}(1, 4)$ — exactly what we expect from $2X+1$.

### 2.2 Multiple Dimensions

For $\mathbf{y} = f(\mathbf{x})$ where $f: \mathbb{R}^n \to \mathbb{R}^n$ is a diffeomorphism:

$$p_Y(\mathbf{y}) = p_X(f^{-1}(\mathbf{y})) \cdot \left|\det\left(\frac{\partial f^{-1}}{\partial \mathbf{y}}\right)\right|$$

The **Jacobian matrix** $J = \frac{\partial f}{\partial \mathbf{x}}$ has entries $J_{ij} = \frac{\partial f_i}{\partial x_j}$.

The **Jacobian determinant** $|\det(J)|$ measures the local volume change factor:
- $|\det(J)| > 1$: the transformation expands volume (density decreases)
- $|\det(J)| < 1$: the transformation compresses volume (density increases)
- $|\det(J)| = 1$: volume-preserving (density unchanged)

---

## 3. Flow Composition

### 3.1 Stacking Transformations

The key insight: compose many simple invertible transformations:

$$\mathbf{x} = f_K \circ f_{K-1} \circ \cdots \circ f_1(\mathbf{z}_0)$$

where $\mathbf{z}_0 \sim p_0(\mathbf{z}_0)$ is the base distribution (typically $\mathcal{N}(0, I)$).

### 3.2 Log-Likelihood Decomposition

The log-likelihood decomposes beautifully:

$$\log p(\mathbf{x}) = \log p_0(\mathbf{z}_0) - \sum_{k=1}^{K} \log \left|\det\left(\frac{\partial f_k}{\partial \mathbf{z}_{k-1}}\right)\right|$$

where $\mathbf{z}_k = f_k(\mathbf{z}_{k-1})$.

**Training:** Maximize this log-likelihood over the parameters of $f_1, ..., f_K$.

**Sampling:** Draw $\mathbf{z}_0 \sim \mathcal{N}(0, I)$, then push forward through $f_K \circ \cdots \circ f_1$.

**Density evaluation:** Map $\mathbf{x}$ backward through $f_1^{-1} \circ \cdots \circ f_K^{-1}$ to get $\mathbf{z}_0$, then compute the formula above.

---

## 4. Types of Flow Transformations

### 4.1 Planar Flows

$$f(\mathbf{z}) = \mathbf{z} + \mathbf{u} \cdot h(\mathbf{w}^T \mathbf{z} + b)$$

where $\mathbf{u}, \mathbf{w} \in \mathbb{R}^n$, $b \in \mathbb{R}$, and $h$ is a smooth activation.

**Jacobian determinant:**
$$\det(J) = 1 + \mathbf{u}^T h'(\mathbf{w}^T \mathbf{z} + b) \mathbf{w}$$

This is $O(n)$ to compute — efficient!

**Limitation:** Each planar flow can only "bend" space along one hyperplane. Many layers needed for complex distributions.

### 4.2 Radial Flows

$$f(\mathbf{z}) = \mathbf{z} + \frac{\beta}{\alpha + r(\mathbf{z})} (\mathbf{z} - \mathbf{z}_0)$$

where $r(\mathbf{z}) = \|\mathbf{z} - \mathbf{z}_0\|$.

These expand or contract space around a reference point $\mathbf{z}_0$.

### 4.3 Coupling Layers (RealNVP)

The breakthrough idea that makes flows practical:

**Split** the input into two halves: $\mathbf{x} = [\mathbf{x}_1, \mathbf{x}_2]$

**Transform:**
$$\mathbf{y}_1 = \mathbf{x}_1 \quad \text{(identity — unchanged!)}$$
$$\mathbf{y}_2 = \mathbf{x}_2 \odot \exp(s(\mathbf{x}_1)) + t(\mathbf{x}_1)$$

where $s$ and $t$ are arbitrary neural networks (scale and translate).

**Why is this brilliant?**

1. **Triangular Jacobian:** The Jacobian is lower-triangular:
$$J = \begin{pmatrix} I & 0 \\ \frac{\partial \mathbf{y}_2}{\partial \mathbf{x}_1} & \text{diag}(\exp(s(\mathbf{x}_1))) \end{pmatrix}$$

2. **Cheap determinant:** $\det(J) = \prod_i \exp(s_i(\mathbf{x}_1)) = \exp(\sum_i s_i(\mathbf{x}_1))$
   This is $O(n)$ instead of $O(n^3)$!

3. **Easy inverse:**
$$\mathbf{x}_1 = \mathbf{y}_1$$
$$\mathbf{x}_2 = (\mathbf{y}_2 - t(\mathbf{y}_1)) \odot \exp(-s(\mathbf{y}_1))$$

4. **Expressive:** $s$ and $t$ can be arbitrarily complex neural networks

**Alternating masks:** To transform ALL dimensions, alternate which half is transformed:
- Layer 1: transform $\mathbf{x}_2$ conditioned on $\mathbf{x}_1$
- Layer 2: transform $\mathbf{x}_1$ conditioned on $\mathbf{x}_2$
- Layer 3: transform $\mathbf{x}_2$ conditioned on $\mathbf{x}_1$
- ...

### 4.4 Autoregressive Flows

**Masked Autoregressive Flow (MAF):**

$$y_i = x_i \cdot \exp(\alpha_i) + \mu_i \quad \text{where} \quad \mu_i, \alpha_i = \text{NN}(x_1, ..., x_{i-1})$$

Each dimension depends only on previous dimensions → lower-triangular Jacobian → $O(n)$ determinant.

**Trade-off:**
- MAF: fast density evaluation, slow sampling (sequential)
- IAF (Inverse Autoregressive Flow): fast sampling, slow density evaluation

---

## 5. Continuous Normalizing Flows

Instead of discrete transformation steps, use an ODE:

$$\frac{d\mathbf{z}(t)}{dt} = f_\theta(\mathbf{z}(t), t)$$

The log-density evolves according to the **instantaneous change of variables:**

$$\frac{d \log p(\mathbf{z}(t))}{dt} = -\text{tr}\left(\frac{\partial f_\theta}{\partial \mathbf{z}(t)}\right)$$

**Advantages:**
- No restriction on architecture (any $f_\theta$)
- Memory-efficient (adjoint method)
- Free-form Jacobian

**Disadvantage:**
- Requires ODE solver during training and sampling (expensive)

This connects directly to **flow matching** (Tutorial 18).

---

## 6. Training Normalizing Flows

### 6.1 Maximum Likelihood

Flows are trained by directly maximizing the log-likelihood:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \log p_\theta(x_i) = \frac{1}{N} \sum_{i=1}^N \left[\log p_0(f^{-1}_\theta(x_i)) + \log\left|\det\left(\frac{\partial f^{-1}_\theta}{\partial x_i}\right)\right|\right]$$

This is a **direct optimization** — no adversarial training, no ELBO, no approximations.

### 6.2 KL Divergence Equivalence

Maximizing likelihood is equivalent to minimizing forward KL divergence:

$$\min_\theta D_{KL}(p_{data} \| p_\theta)$$

This is **mode-covering** (unlike GANs which are mode-seeking), meaning flows tend to spread probability to cover all modes of the data.

---

## 7. Universal Approximation

**Theorem (flow universality):** Under mild conditions, normalizing flows with coupling layers can approximate any target distribution arbitrarily well, given enough layers.

However, practical limitations exist:
- Finite depth means limited expressiveness
- Topology preservation: flows cannot change the topology (e.g., can't map a connected set to disconnected modes without enough layers)
- The required number of layers may be very large

---

## 8. Comparison with Other Generative Models

| Feature | Flows | GANs | VAEs | Diffusion |
|---------|-------|------|------|-----------|
| Density evaluation | **Exact** | None | ELBO | Approximate |
| Sample quality | Good | Excellent | Fair | Excellent |
| Training stability | **Stable** (MLE) | Unstable | Stable | Stable |
| Sampling speed | Fast (single pass) | **Fast** | Fast | Slow |
| Architecture constraints | **Invertible** | None | Encoder+Decoder | None |
| Mode coverage | **Full** | Partial | Full | Full |
| Latent space | **Meaningful** | None | Meaningful | None |

### When to use flows:
- When you need **exact likelihood** (anomaly detection, model comparison)
- When you need a **meaningful, invertible** latent space
- When training stability matters more than sample quality
- When you need both sampling AND density evaluation

---

## 9. Notable Flow Architectures

| Model | Year | Key Contribution |
|-------|------|-----------------|
| **NICE** | 2014 | First coupling layer flow |
| **RealNVP** | 2016 | Affine coupling with scale |
| **Glow** | 2018 | 1×1 convolutions, invertible |
| **FFJORD** | 2019 | Free-form continuous flows |
| **Neural Spline Flows** | 2019 | Spline-based coupling layers |
| **Flow++** | 2019 | Improved dequantization |

---

## 10. Connection to Information Theory

From our information-theoretic perspective:

- **Base distribution** $p_0$ = maximum entropy prior (Gaussian)
- **Flow transformations** = structured removal of entropy from the latent space
- **Training** = finding the transformation that makes data look most Gaussian when mapped backward
- **Log-determinant** = measuring how much the flow concentrates or spreads probability
- **KL minimization** = maximizing the information the model captures about the data

The flow literally "normalizes" complex distributions back to the simple Gaussian — hence the name "normalizing flow."

---

## Summary of Key Equations

| Concept | Equation |
|---------|----------|
| Change of variables (1D) | $p_Y(y) = p_X(f^{-1}(y)) \cdot \|df^{-1}/dy\|$ |
| Change of variables (nD) | $p_Y(y) = p_X(f^{-1}(y)) \cdot \|\det(J^{-1})\|$ |
| Log-likelihood | $\log p(x) = \log p_0(z_0) - \sum_k \log\|\det(J_k)\|$ |
| Coupling layer | $y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)$ |
| Coupling Jacobian | $\det(J) = \exp(\sum_i s_i(x_1))$ |
| Continuous flow | $dz/dt = f_\theta(z, t)$, $d\log p/dt = -\text{tr}(\partial f/\partial z)$ |
