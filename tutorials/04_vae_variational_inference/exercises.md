# Tutorial 04: VAE & Variational Inference — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Why Latent Variables?
Explain why we need latent variables. What problem do they solve in generative modeling?

### Exercise A2 🟡 — ELBO Derivation
Starting from $\log p(x)$, derive the Evidence Lower Bound (ELBO):
$$\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))$$

### Exercise A3 🟡 — ELBO as Reconstruction + Regularization
Interpret each term in the ELBO:
1. $E_{q(z|x)}[\log p(x|z)]$
2. $-D_{KL}(q(z|x) || p(z))$

### Exercise A4 🔴 — Reparameterization Trick
Why can't we directly backpropagate through $z \sim q(z|x)$? Derive the reparameterization trick for a Gaussian $q(z|x) = N(\mu(x), \sigma^2(x))$.

### Exercise A5 🔴 — KL for Gaussian Posterior
In a standard VAE with $p(z) = N(0, I)$ and $q(z|x) = N(\mu, \text{diag}(\sigma^2))$, derive:
$$D_{KL}(q || p) = \frac{1}{2}\sum_{j=1}^{d}\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

### Exercise A6 🔴 — Posterior Collapse
What is "posterior collapse" in VAEs? Why does it happen and how can it be mitigated?

---

## Part B: Coding

### Exercise B1 🟡 — Implement Reparameterization
```python
# TODO: Implement the reparameterization trick
# 1. Given mu and log_var from encoder
# 2. Sample z using reparameterization
# 3. Verify gradients flow through
```

### Exercise B2 🟡 — KL Loss Implementation
```python
# TODO: Implement KL divergence loss for VAE
# 1. KL(N(μ, σ²) || N(0, 1))
# 2. Handle batched inputs
# 3. Sum over latent dimensions
```

### Exercise B3 🔴 — Full VAE Implementation
```python
# TODO: Implement a complete VAE in PyTorch
# 1. Encoder: x → (μ, log σ²)
# 2. Reparameterization: sample z
# 3. Decoder: z → x̂
# 4. Loss: reconstruction + KL
# 5. Train on MNIST
# 6. Visualize latent space and reconstructions
```

---

## Part C: Conceptual

### C1 🟡
Why do we use $\log\sigma^2$ instead of $\sigma$ as the encoder output?

### C2 🔴
Compare VAE to standard autoencoder. Why does the KL term encourage meaningful latent representations?
