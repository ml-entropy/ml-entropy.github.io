# Tutorial 16: Exercises — Generative Adversarial Networks

---

## Part A: Theory

### Exercise A1: Optimal Discriminator Derivation 🟢 Easy

Given the GAN value function:

$$V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**(a)** Rewrite $V(D, G)$ as an integral over $x$ involving $p_{data}(x)$ and $p_g(x)$.

**(b)** For a fixed $x$, find the value of $D(x)$ that maximizes $p_{data}(x) \log D(x) + p_g(x) \log(1 - D(x))$.

**(c)** Verify that the second derivative is negative (confirming it's a maximum).

**(d)** What is $D^*(x)$ when $p_{data}(x) = p_g(x)$? Interpret this result.

---

### Exercise A2: Jensen-Shannon Divergence Connection 🟡 Medium

**(a)** Starting from $V(D^*, G)$, substitute the optimal discriminator and show that:

$$V(D^*, G) = -\log 4 + 2 \cdot JSD(p_{data} \| p_g)$$

*Hint: Define $m(x) = \frac{p_{data}(x) + p_g(x)}{2}$ and use the definition of KL divergence.*

**(b)** Why is the JSD bounded between 0 and $\log 2$, while KL divergence is unbounded?

**(c)** Prove that $V(D^*, G) \geq -\log 4$ with equality iff $p_g = p_{data}$.

---

### Exercise A3: Non-Saturating Loss Analysis 🟡 Medium

In practice, instead of minimizing $\mathbb{E}[\log(1 - D(G(z)))]$, the generator maximizes $\mathbb{E}[\log D(G(z))]$.

**(a)** Compute the gradient $\frac{\partial}{\partial D(G(z))} \log(1 - D(G(z)))$ when $D(G(z)) \approx 0$ (early training). What happens?

**(b)** Compute the gradient $\frac{\partial}{\partial D(G(z))} \log D(G(z))$ when $D(G(z)) \approx 0$. How does it compare?

**(c)** Show that both objectives have the same fixed point (i.e., both are optimized when $p_g = p_{data}$).

**(d)** What divergence does the non-saturating loss correspond to? *Hint: It involves reverse KL.*

---

### Exercise A4: Mode Collapse Analysis 🟡 Medium

Consider a data distribution that is a mixture of two Gaussians: $p_{data} = 0.5 \mathcal{N}(-3, 0.5) + 0.5 \mathcal{N}(3, 0.5)$.

**(a)** If the generator collapses to producing only $x = 3$ (i.e., $p_g = \delta_3$), what is $D^*(x)$?

**(b)** Compute $JSD(p_{data} \| p_g)$ for this collapsed generator. Compare to $JSD$ when $p_g = p_{data}$.

**(c)** Explain intuitively why the generator might prefer to capture only one mode. *Hint: Think about the risk of generating samples between the modes.*

**(d)** How does the KL divergence direction matter? Compare $D_{KL}(p_g \| p_{data})$ vs $D_{KL}(p_{data} \| p_g)$ — which encourages mode coverage?

---

### Exercise A5: Wasserstein Distance Properties 🟡 Medium

**(a)** For $P = \delta_0$ and $Q = \delta_\theta$, compute:
  - $JSD(P \| Q)$ for $\theta \neq 0$
  - $W_1(P, Q)$

**(b)** Compute $\frac{\partial}{\partial \theta} JSD(P \| Q)$ and $\frac{\partial}{\partial \theta} W_1(P, Q)$. Which provides a useful gradient signal?

**(c)** Explain the 1-Lipschitz constraint intuitively. Why is it necessary for the Wasserstein distance?

**(d)** In the gradient penalty approach, interpolated samples $\hat{x} = \alpha x_{real} + (1-\alpha) x_{fake}$ are used. Why sample along interpolations rather than from the data or generator alone?

---

### Exercise A6: f-Divergences and GANs 🔴 Hard

The f-divergence between distributions $P$ and $Q$ is:

$$D_f(P \| Q) = \mathbb{E}_{x \sim Q}\left[f\left(\frac{p(x)}{q(x)}\right)\right]$$

where $f$ is a convex function with $f(1) = 0$.

**(a)** Show that $f(t) = t \log t$ gives the KL divergence $D_{KL}(P \| Q)$.

**(b)** Show that $f(t) = -\log t$ gives the reverse KL divergence $D_{KL}(Q \| P)$.

**(c)** What $f$ gives the Jensen-Shannon divergence?

**(d)** Using the Fenchel conjugate $f^*(u) = \sup_t [ut - f(t)]$, derive a variational lower bound on $D_f(P \| Q)$ that can be estimated with samples. This is the basis for f-GANs.

---

## Part B: Coding

### Exercise B1: Implement the Optimal Discriminator 🟢 Easy

```python
import numpy as np
import matplotlib.pyplot as plt

def optimal_discriminator(x, p_data_fn, p_g_fn):
    """
    Compute the optimal discriminator D*(x).

    Args:
        x: array of points
        p_data_fn: function computing p_data(x)
        p_g_fn: function computing p_g(x)

    Returns:
        D*(x) for each point
    """
    # TODO: Implement D*(x) = p_data(x) / (p_data(x) + p_g(x))
    pass

# Define distributions
# p_data = N(0, 1), p_g = N(2, 1.5)
# (a) Plot both distributions and D*(x) on the same figure
# (b) At what point does D*(x) = 0.5? Explain why.
# (c) Repeat with p_g = N(0, 1) (generator perfectly matches data).
#     What does D*(x) look like?
```

---

### Exercise B2: Visualize Training Dynamics 🟡 Medium

```python
# Simulate GAN training dynamics for 1D distributions
#
# Real data: mixture of 2 Gaussians at -3 and +3
# Generator: starts as a single Gaussian, parameterized by (mu, sigma)
#
# (a) For each training step:
#     1. Compute D*(x) given current p_data and p_g
#     2. Compute the gradient of the generator loss w.r.t. mu and sigma
#     3. Update mu and sigma with gradient descent
#     4. Record JSD(p_data || p_g)
#
# (b) Plot:
#     - p_data and p_g at training steps 0, 25, 50, 100
#     - D*(x) at the same steps
#     - JSD over training
#
# (c) Does the generator converge? Does it capture both modes?
#     If not, explain why.
```

---

### Exercise B3: Mode Collapse Demonstration 🟡 Medium

```python
# Implement a simple GAN (numpy only) that exhibits mode collapse
#
# Real data: mixture of 4 Gaussians at (-3,-3), (-3,3), (3,-3), (3,3)
# Generator: MLP with 1 hidden layer (noise_dim -> 32 -> 2)
# Discriminator: MLP with 1 hidden layer (2 -> 32 -> 1)
#
# (a) Train the GAN for 2000 steps. At steps 0, 500, 1000, 2000:
#     - Plot real data and generated data side by side
#     - Count how many modes the generator covers
#
# (b) Record and plot D_loss and G_loss over training.
#     Do they converge or oscillate?
#
# (c) Try different G:D training ratios (1:1, 1:5, 5:1).
#     Which shows the worst mode collapse? Why?
```

---

### Exercise B4: Distance Metric Comparison 🟡 Medium

```python
from scipy.stats import wasserstein_distance

# Compare JSD and Wasserstein distance behavior
#
# (a) Create two Gaussians: P = N(0, 1) and Q = N(d, 1)
#     For d in [0, 0.5, 1, 2, 3, 5, 10]:
#     - Compute JSD(P, Q)
#     - Compute W_1(P, Q) using scipy
#     Plot both on the same axes. How do they scale with d?
#
# (b) Create two narrow Gaussians: P = N(0, 0.01) and Q = N(d, 0.01)
#     Repeat the comparison. When the distributions barely overlap,
#     what happens to JSD? What about W_1?
#
# (c) Simulate the "sliding distributions" scenario:
#     P = N(0, 1), Q = N(d, 1) for d from -5 to 5.
#     Plot JSD and W_1 as functions of d.
#     Which provides smoother gradients?
```

---

### Exercise B5: Implement WGAN Gradient Penalty 🔴 Hard

```python
# Implement WGAN-GP for 2D data
#
# Real data: two moons (generate manually or use sklearn)
# Critic: MLP (2 -> 64 -> 64 -> 1) — NO sigmoid!
# Generator: MLP (noise_dim -> 64 -> 64 -> 2)
#
# (a) Implement the critic loss:
#     L_critic = E[D(x_fake)] - E[D(x_real)] + λ * gradient_penalty
#     where gradient_penalty = E[(||∇D(x̂)||₂ - 1)²]
#     and x̂ = αx_real + (1-α)x_fake
#
# (b) Implement the generator loss:
#     L_gen = -E[D(x_fake)]
#
# (c) Train for 5000 steps with:
#     - 5 critic steps per generator step
#     - λ = 10
#     - Learning rate = 1e-4
#
# (d) Plot:
#     - Generated samples vs real data
#     - Critic loss over training (should be meaningful, unlike standard GAN!)
#     - The critic function as a contour plot
#
# (e) Compare with standard GAN training on the same data.
#     Is WGAN-GP more stable?
```
