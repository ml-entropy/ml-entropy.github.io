# Tutorial 02: KL Divergence — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — KL Divergence Calculation
Given $P = [0.6, 0.3, 0.1]$ and $Q = [0.5, 0.4, 0.1]$:
a) Calculate $D_{KL}(P || Q)$
b) Calculate $D_{KL}(Q || P)$
c) Are they equal? What does this tell you?

### Exercise A2 🟢 — When is KL Zero?
Under what conditions is $D_{KL}(P || Q) = 0$? Prove it.

### Exercise A3 🟡 — Cross-Entropy Decomposition
Prove: $H(P, Q) = H(P) + D_{KL}(P || Q)$

### Exercise A4 🟡 — KL and Jensen's Inequality
Prove $D_{KL}(P || Q) \geq 0$ using Jensen's inequality on $\log$.

### Exercise A5 🔴 — KL Between Gaussians
Derive the KL divergence between two univariate Gaussians:
$$D_{KL}(N(\mu_1, \sigma_1^2) || N(\mu_2, \sigma_2^2))$$

### Exercise A6 🔴 — Forward vs Reverse KL
Explain why minimizing $D_{KL}(Q || P)$ (forward) is mode-covering while $D_{KL}(P || Q)$ (reverse) is mode-seeking. Draw diagrams.

---

## Part B: Coding

### Exercise B1 🟢 — KL Calculator
```python
# TODO: Implement KL divergence
# 1. Handle P(x)=0 correctly
# 2. Detect and handle Q(x)=0 when P(x)>0 (should be infinity)
# 3. Test on known distributions
```

### Exercise B2 🟡 — KL as Loss Function
```python
# TODO: Train a model to approximate distribution P using KL loss
# 1. Generate samples from mixture of Gaussians
# 2. Fit a single Gaussian by minimizing KL
# 3. Compare forward vs reverse KL results
```

### Exercise B3 🔴 — KL Between Empirical Distributions
```python
# TODO: Estimate KL from samples
# 1. Generate samples from P and Q
# 2. Use histogram-based KL estimation
# 3. Compare to analytical KL (if known)
# 4. Analyze bias and variance vs sample size
```

---

## Part C: Conceptual

### C1 🟡
Why can't we use KL divergence as a distance metric?

### C2 🔴
In VAEs, we minimize $D_{KL}(q(z|x) || p(z))$. Why this direction and not the reverse?
