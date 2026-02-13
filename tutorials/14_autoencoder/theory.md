# Tutorial 13: Autoencoders — Learning Compressed Representations

## Overview
Autoencoders learn to compress data into a lower-dimensional representation and reconstruct it. We derive the architecture, understand the information bottleneck, and explore variants: sparse, denoising, and contractive autoencoders. This sets the stage for variational autoencoders.

---

## Part 1: Why Autoencoders?

### The Representation Problem

Raw data (images, text, audio) lives in very high-dimensional spaces. But the **intrinsic dimensionality** is often much lower:

- A 64x64 grayscale image has 4,096 dimensions
- But the space of "meaningful" images is a tiny manifold within $\mathbb{R}^{4096}$

**Goal:** Learn a function that maps data to a compact, useful representation.

### The Autoencoder Idea

An autoencoder is trained to **copy its input to its output** through a bottleneck:

```
Input x ∈ ℝⁿ
    ↓
[Encoder f_θ] → z ∈ ℝᵈ    (d << n)
    ↓
[Decoder g_φ] → x̂ ∈ ℝⁿ
    ↓
Minimize ||x - x̂||²
```

The bottleneck forces the network to learn what matters.

### Why Not Just Use PCA?

PCA finds the best **linear** low-dimensional projection. Autoencoders can learn **nonlinear** manifolds:

| Property | PCA | Autoencoder |
|----------|-----|-------------|
| Mapping | Linear | Nonlinear |
| Optimality | Global (for linear) | Local (gradient-based) |
| Capacity | Fixed by eigenvalues | Flexible architecture |
| Interpretability | Orthogonal axes | Learned features |

**Key result:** A single-layer autoencoder with linear activations learns the same subspace as PCA.

---

## Part 2: The Mathematics

### Formal Definition

**Encoder:**
$$z = f_\theta(x) = \sigma(Wx + b)$$

**Decoder:**
$$\hat{x} = g_\phi(z) = \sigma'(W'z + b')$$

**Loss function:**
$$\mathcal{L}(\theta, \phi) = \frac{1}{N}\sum_{i=1}^{N} \|x_i - g_\phi(f_\theta(x_i))\|^2$$

### The Reconstruction Objective

For continuous data (MSE loss):
$$\mathcal{L}_{MSE} = \|x - \hat{x}\|^2 = \sum_{j=1}^n (x_j - \hat{x}_j)^2$$

For binary data (binary cross-entropy loss):
$$\mathcal{L}_{BCE} = -\sum_{j=1}^n \left[x_j \log \hat{x}_j + (1 - x_j) \log(1 - \hat{x}_j)\right]$$

### Why BCE for Binary Data?

If we model the decoder output as $p(x_j = 1 | z) = \hat{x}_j$, then:

$$\log p(x | z) = \sum_j \left[x_j \log \hat{x}_j + (1 - x_j)\log(1 - \hat{x}_j)\right]$$

Minimizing $-\log p(x|z)$ gives BCE. This is the **maximum likelihood** perspective.

### Deep Autoencoders

Stack multiple layers for more expressive encodings:

**Encoder:**
$$h_1 = \sigma(W_1 x + b_1)$$
$$h_2 = \sigma(W_2 h_1 + b_2)$$
$$z = \sigma(W_3 h_2 + b_3)$$

**Decoder (mirror architecture):**
$$h_2' = \sigma(W_3' z + b_3')$$
$$h_1' = \sigma(W_2' h_2' + b_2')$$
$$\hat{x} = \sigma'(W_1' h_1' + b_1')$$

**Tied weights variant:** Set $W_i' = W_i^T$ to halve the parameters and act as regularization.

---

## Part 3: The Information Bottleneck Perspective

### Autoencoders as Compression

The encoder performs **lossy compression**:

$$\text{Input } x \xrightarrow{\text{encode}} z \xrightarrow{\text{decode}} \hat{x} \approx x$$

With $d < n$, information must be lost. The autoencoder learns **which information to keep**.

### Connection to Information Theory

Define the mutual information between input and latent code:

$$I(X; Z) = H(X) - H(X | Z)$$

- **$H(X)$** is fixed (property of the data)
- **$H(X|Z)$** measures remaining uncertainty after encoding
- Minimizing reconstruction loss $\approx$ minimizing $H(X|Z)$

Therefore, training maximizes $I(X; Z)$ subject to the bottleneck constraint $\dim(Z) = d$.

### Rate-Distortion Theory

From Shannon's rate-distortion theory:

$$R(D) = \min_{p(z|x): \mathbb{E}[d(x,\hat{x})] \leq D} I(X; Z)$$

The autoencoder approximates this optimal trade-off:
- **Rate**: Bits needed to represent $z$ (limited by bottleneck dimension)
- **Distortion**: Reconstruction error $\|x - \hat{x}\|^2$

---

## Part 4: Undercomplete vs Overcomplete Autoencoders

### Undercomplete ($d < n$)

When the latent dimension is smaller than the input, the bottleneck alone forces compression:

$$z \in \mathbb{R}^d, \quad d < n$$

- **Advantage:** Simple, guaranteed to learn compressed representation
- **Risk:** May be too restrictive; not enough capacity to capture important features

### Overcomplete ($d \geq n$)

When the latent dimension equals or exceeds the input:

$$z \in \mathbb{R}^d, \quad d \geq n$$

**Problem:** The network can learn the identity function $f(x) = x$ with zero loss!

**Solution:** Add explicit regularization to prevent trivial solutions. This leads to the variants below.

---

## Part 5: Sparse Autoencoders

### The Idea

Add a sparsity penalty so most latent units are inactive for any given input:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \sum_{j=1}^d |z_j|$$

This is the **L1 penalty** on activations (not weights).

### KL Divergence Sparsity

For sigmoid activations, define the average activation:
$$\hat{\rho}_j = \frac{1}{N}\sum_{i=1}^N z_j^{(i)}$$

We want $\hat{\rho}_j \approx \rho$ for a target sparsity $\rho$ (e.g., 0.05).

**KL sparsity penalty:**
$$\Omega_{sparse} = \sum_{j=1}^d D_{KL}(\rho \| \hat{\rho}_j) = \sum_{j=1}^d \left[\rho \log \frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}\right]$$

### Why Sparsity?

1. **Overcomplete representations** become useful (each input activates a different subset)
2. **Interpretable features**: Each unit learns a distinct pattern
3. **Biological plausibility**: Neurons in cortex have sparse firing patterns
4. **Connection to dictionary learning**: Sparse codes = sparse linear combinations of dictionary atoms

---

## Part 6: Denoising Autoencoders

### The Idea

Train the autoencoder to reconstruct **clean** input from **corrupted** input:

1. Corrupt input: $\tilde{x} = \text{corrupt}(x)$
2. Encode: $z = f_\theta(\tilde{x})$
3. Decode: $\hat{x} = g_\phi(z)$
4. Loss: $\|x - \hat{x}\|^2$ (compare to **original** $x$, not corrupted $\tilde{x}$)

### Common Corruption Strategies

**Masking noise:** Set random subset of inputs to zero
$$\tilde{x}_j = \begin{cases} x_j & \text{with probability } 1 - p \\ 0 & \text{with probability } p \end{cases}$$

**Gaussian noise:** Add random noise
$$\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

**Salt-and-pepper noise:** Randomly set pixels to 0 or 1

### Why Does Denoising Help?

**Theorem (Vincent et al., 2010):** Minimizing the denoising reconstruction error is equivalent to learning a vector field that points toward the data manifold.

More precisely, the optimal denoising function satisfies:
$$g^*({\tilde{x}}) = \mathbb{E}[X | \tilde{X} = \tilde{x}]$$

This is the **conditional expectation**, which projects corrupted points back onto the data manifold.

### Score Matching Connection

For Gaussian noise with small $\sigma$:
$$\nabla_{\tilde{x}} \log p(\tilde{x}) \approx \frac{g^*(\tilde{x}) - \tilde{x}}{\sigma^2}$$

The denoising autoencoder implicitly learns the **score function** (gradient of log-density). This is the foundation of modern **diffusion models**.

---

## Part 7: Contractive Autoencoders

### The Idea

Penalize the encoder for being sensitive to input perturbations:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \left\|\frac{\partial f_\theta(x)}{\partial x}\right\|_F^2$$

where $\|\cdot\|_F$ is the Frobenius norm of the Jacobian.

### The Jacobian Penalty

The Jacobian $J = \frac{\partial z}{\partial x} \in \mathbb{R}^{d \times n}$ measures how much $z$ changes when $x$ changes:

$$\|J\|_F^2 = \sum_{i=1}^d \sum_{j=1}^n \left(\frac{\partial z_i}{\partial x_j}\right)^2$$

Minimizing this makes $z$ **locally invariant** to input perturbations.

### Why Contractiveness?

- Pulls the representation toward a **lower-dimensional manifold**
- The encoder learns to ignore directions of no variation
- Combines the benefits of denoising (robust to noise) with explicit regularization

### Comparison of Regularized Autoencoders

| Variant | Regularizer | Effect |
|---------|-------------|--------|
| Sparse | $\lambda \sum|z_j|$ | Few active units per input |
| Denoising | Corrupt input | Learn to denoise; score matching |
| Contractive | $\lambda \|J\|_F^2$ | Locally invariant encodings |

---

## Part 8: Training Autoencoders in Practice

### Architecture Design

**Encoder:** Gradually reduce dimensions
$$n \to 512 \to 256 \to 128 \to d$$

**Decoder:** Mirror the encoder
$$d \to 128 \to 256 \to 512 \to n$$

**For images:** Use convolutional layers (Conv encoder, ConvTranspose decoder)

### Activation Functions

| Layer | Typical Activation | Why |
|-------|-------------------|-----|
| Encoder hidden | ReLU or LeakyReLU | Good gradients, sparse activations |
| Latent | None (linear) | Unrestricted representation |
| Decoder hidden | ReLU or LeakyReLU | Match encoder |
| Decoder output | Sigmoid (binary) or None (continuous) | Match data range |

### Common Pitfalls

1. **Latent dimension too large:** Network learns identity
2. **Latent dimension too small:** Underfitting, blurry reconstructions
3. **Decoder too powerful:** Ignores latent code
4. **No regularization with overcomplete:** Trivial solutions

### Choosing the Latent Dimension

Rule of thumb: Start with $d \approx \sqrt{n}$ and adjust based on:
- Reconstruction quality (too blurry → increase $d$)
- Downstream task performance
- Visualization of latent space (look for structure)

---

## Part 9: Applications

### Dimensionality Reduction

Train autoencoder, use encoder output $z$ for:
- Visualization (2D or 3D latent space)
- Clustering (k-means on $z$)
- Classification (train classifier on $z$)

### Anomaly Detection

Train autoencoder on "normal" data. At test time:
$$\text{anomaly score}(x) = \|x - g_\phi(f_\theta(x))\|^2$$

High reconstruction error → anomalous input.

**Why this works:** The autoencoder learns to reconstruct the training distribution. Inputs from a different distribution will be poorly reconstructed.

### Pretraining for Deep Networks

Historically, layer-wise pretraining with autoencoders was crucial:
1. Train autoencoder on raw data → get first layer features
2. Train autoencoder on first layer features → get second layer features
3. Fine-tune entire stack with supervised objective

This is less common now (thanks to better initialization and BatchNorm), but the principle of learning representations remains central.

---

## Part 10: From Autoencoders to VAEs — A Preview

### The Limitation

Standard autoencoders learn deterministic encodings:
$$z = f_\theta(x)$$

**Problems:**
- Latent space may have "holes" (no training data mapped there)
- Can't sample meaningful new data from the latent space
- No probabilistic interpretation

### The VAE Solution (Next Tutorial)

VAEs make the encoding **probabilistic**:
$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$$

And add a KL divergence penalty to regularize the latent space:
$$\mathcal{L}_{VAE} = \text{Reconstruction} + D_{KL}(q(z|x) \| p(z))$$

This gives:
- A smooth, continuous latent space
- The ability to generate new samples by sampling $z \sim p(z)$
- A principled probabilistic framework (variational inference)

### The Key Insight

| Autoencoder | VAE |
|-------------|-----|
| $z = f(x)$ (point) | $z \sim q(z|x)$ (distribution) |
| Bottleneck regularizes | KL divergence regularizes |
| Can't generate | Can generate |
| Reconstruction only | Reconstruction + KL |

---

## Summary

### Key Takeaways

1. **Autoencoders learn compressed representations** by reconstructing input through a bottleneck
2. **The bottleneck forces** the network to learn what information is essential
3. **Information theory** provides the framework: maximize $I(X; Z)$ given limited capacity
4. **Regularization variants** (sparse, denoising, contractive) prevent trivial solutions and improve learned features
5. **Denoising autoencoders** implicitly learn the score function — connecting to modern diffusion models
6. **Standard autoencoders can't generate** — motivating the variational approach (next tutorial)

### Key Formulas

| Concept | Formula |
|---------|---------|
| Reconstruction loss (MSE) | $\mathcal{L} = \|x - \hat{x}\|^2$ |
| Reconstruction loss (BCE) | $\mathcal{L} = -\sum_j [x_j \log \hat{x}_j + (1-x_j)\log(1-\hat{x}_j)]$ |
| Sparse penalty (L1) | $\Omega = \lambda \sum_j |z_j|$ |
| Sparse penalty (KL) | $\Omega = \sum_j D_{KL}(\rho \| \hat{\rho}_j)$ |
| Contractive penalty | $\Omega = \lambda \|J_f(x)\|_F^2$ |
| Anomaly score | $s(x) = \|x - g(f(x))\|^2$ |

---

## Further Reading

1. **Original deep autoencoders:** Hinton & Salakhutdinov, "Reducing the Dimensionality of Data with Neural Networks" (2006)
2. **Denoising autoencoders:** Vincent et al., "Extracting and Composing Robust Features with Denoising Autoencoders" (2008)
3. **Sparse autoencoders:** Andrew Ng, CS294A Lecture Notes (2011)
4. **Contractive autoencoders:** Rifai et al., "Contractive Auto-Encoders" (2011)
5. **Score matching connection:** Vincent, "A Connection Between Score Matching and Denoising Autoencoders" (2011)
