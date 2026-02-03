# Tutorial 02: KL Divergence — Solutions

---

## Part A: Theory Solutions

### Solution A1 — KL Divergence Calculation

$P = [0.6, 0.3, 0.1]$, $Q = [0.5, 0.4, 0.1]$

a) $D_{KL}(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$
$$= 0.6 \log\frac{0.6}{0.5} + 0.3 \log\frac{0.3}{0.4} + 0.1 \log\frac{0.1}{0.1}$$
$$= 0.6 \times 0.182 + 0.3 \times (-0.288) + 0.1 \times 0$$
$$= 0.109 - 0.086 + 0 = 0.023 \text{ nats}$$

b) $D_{KL}(Q || P) = \sum_i Q(i) \log \frac{Q(i)}{P(i)}$
$$= 0.5 \log\frac{0.5}{0.6} + 0.4 \log\frac{0.4}{0.3} + 0.1 \times 0$$
$$= 0.5 \times (-0.182) + 0.4 \times 0.288 = -0.091 + 0.115 = 0.024 \text{ nats}$$

c) **No, they're not equal!** KL divergence is asymmetric. This is why it's not a true distance metric.

---

### Solution A2 — When is KL Zero?

$D_{KL}(P || Q) = 0$ **if and only if** $P = Q$ (distributions are identical).

**Proof**:
- If $P = Q$: $D_{KL} = \sum P(i) \log 1 = 0$ ✓
- If $P \neq Q$: Since $-\log$ is strictly convex and we have $D_{KL} \geq 0$ with equality only when $P(i)/Q(i) = 1$ for all $i$, we need $P = Q$.

---

### Solution A3 — Cross-Entropy Decomposition

$$H(P, Q) = -\sum_i P(i) \log Q(i)$$

Add and subtract $\log P(i)$:
$$= -\sum_i P(i) \log Q(i) + \sum_i P(i) \log P(i) - \sum_i P(i) \log P(i)$$
$$= -\sum_i P(i) \log P(i) + \sum_i P(i) \log \frac{P(i)}{Q(i)}$$
$$= H(P) + D_{KL}(P || Q)$$ ∎

---

### Solution A4 — KL Non-Negativity via Jensen

$$D_{KL}(P || Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} = -\sum_i P(i) \log \frac{Q(i)}{P(i)}$$

Since $-\log$ is convex, by Jensen's inequality:
$$-\sum_i P(i) \log \frac{Q(i)}{P(i)} \geq -\log \sum_i P(i) \frac{Q(i)}{P(i)} = -\log \sum_i Q(i) = -\log 1 = 0$$

Therefore $D_{KL}(P || Q) \geq 0$. ∎

---

### Solution A5 — KL Between Gaussians

Let $p(x) = N(\mu_1, \sigma_1^2)$, $q(x) = N(\mu_2, \sigma_2^2)$.

$$D_{KL} = \int p(x) \log \frac{p(x)}{q(x)} dx = E_p[\log p - \log q]$$

$$\log p = -\frac{1}{2}\log(2\pi\sigma_1^2) - \frac{(x-\mu_1)^2}{2\sigma_1^2}$$
$$\log q = -\frac{1}{2}\log(2\pi\sigma_2^2) - \frac{(x-\mu_2)^2}{2\sigma_2^2}$$

$$D_{KL} = \frac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} + E_p\left[\frac{(x-\mu_2)^2}{2\sigma_2^2} - \frac{(x-\mu_1)^2}{2\sigma_1^2}\right]$$

Using $E_p[(x-\mu_1)^2] = \sigma_1^2$ and $E_p[(x-\mu_2)^2] = \sigma_1^2 + (\mu_1-\mu_2)^2$:

$$\boxed{D_{KL} = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}}$$

---

### Solution A6 — Forward vs Reverse KL

**Forward KL** $D_{KL}(P || Q)$: Forces Q to cover all of P's support
- Where $P(x) > 0$, we need $Q(x) > 0$ (else infinite penalty)
- Results in **mode-covering**: Q spreads to cover all modes of P

**Reverse KL** $D_{KL}(Q || P)$: Allows Q to be selective
- Only penalized where $Q(x) > 0$
- Can ignore modes of P entirely
- Results in **mode-seeking**: Q focuses on one mode

---

## Part B: Coding Solutions

### Solution B1 — KL Calculator

```python
import numpy as np

def kl_divergence(p, q, eps=1e-10):
    """
    Compute KL(P || Q).
    
    Returns infinity if Q(x)=0 where P(x)>0.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Check for invalid case: P(x) > 0 but Q(x) = 0
    if np.any((p > eps) & (q < eps)):
        return np.inf
    
    # Compute KL only where P > 0
    mask = p > eps
    return np.sum(p[mask] * np.log(p[mask] / (q[mask] + eps)))

# Test
P = np.array([0.6, 0.3, 0.1])
Q = np.array([0.5, 0.4, 0.1])
print(f"KL(P||Q) = {kl_divergence(P, Q):.4f}")
print(f"KL(Q||P) = {kl_divergence(Q, P):.4f}")
print(f"KL(P||P) = {kl_divergence(P, P):.4f}")  # Should be 0
```

### Solution B2 — KL as Loss Function

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# True distribution: mixture of two Gaussians
def true_pdf(x):
    return 0.5 * stats.norm.pdf(x, -2, 0.8) + 0.5 * stats.norm.pdf(x, 2, 0.8)

# Fit single Gaussian by minimizing forward KL (mode-covering)
# KL(P||Q) ≈ E_P[-log Q] + const = E_P[(x-μ)²/2σ²] + log σ + const
# Closed form: μ = E[X], σ² = Var[X] under P

# Generate samples from P
np.random.seed(42)
samples = np.concatenate([
    np.random.normal(-2, 0.8, 5000),
    np.random.normal(2, 0.8, 5000)
])

# Forward KL solution (moment matching)
mu_forward = np.mean(samples)  # ≈ 0
sigma_forward = np.std(samples)  # ≈ 2.4

# Reverse KL: mode-seeking (would collapse to one mode)
# Approximate by finding mode with gradient descent
mu_reverse = -2  # One of the modes
sigma_reverse = 0.8

# Plot
x = np.linspace(-6, 6, 200)
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, true_pdf(x), 'b-', linewidth=2, label='True P (mixture)')
ax.plot(x, stats.norm.pdf(x, mu_forward, sigma_forward), 'r--', linewidth=2, 
        label=f'Forward KL fit (μ={mu_forward:.1f}, σ={sigma_forward:.1f})')
ax.plot(x, stats.norm.pdf(x, mu_reverse, sigma_reverse), 'g:', linewidth=2,
        label=f'Reverse KL fit (μ={mu_reverse:.1f}, σ={sigma_reverse:.1f})')

ax.set_xlabel('x')
ax.set_ylabel('Density')
ax.set_title('Forward KL (mode-covering) vs Reverse KL (mode-seeking)')
ax.legend()
plt.show()

print("Forward KL: covers both modes but blurs between them")
print("Reverse KL: focuses on one mode, ignores the other")
```

---

## Part C: Conceptual Solutions

### C1
KL divergence is not a metric because:
1. **Asymmetric**: $D_{KL}(P||Q) \neq D_{KL}(Q||P)$
2. **No triangle inequality**: Can have $D_{KL}(P||R) > D_{KL}(P||Q) + D_{KL}(Q||R)$

### C2
In VAEs, we minimize $D_{KL}(q(z|x) || p(z))$ (reverse KL) because:
1. **Computable**: We can sample from $q$ (encoder) and evaluate $p$ (prior)
2. **Forward KL would require**: Sampling from prior and evaluating encoder — doesn't use the data!
3. **Mode-seeking is OK**: We want the approximate posterior to focus on regions of high probability under the prior
