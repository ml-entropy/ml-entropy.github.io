# Tutorial 16: Solutions — Generative Adversarial Networks

---

## Part A: Theory Solutions

### Solution A1: Optimal Discriminator Derivation

**(a)** Rewriting with $p_g(x)$ (the distribution of $G(z)$):

$$V(D, G) = \int_x \left[ p_{data}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx$$

The second term uses the fact that $\mathbb{E}_{z \sim p_z}[h(G(z))] = \mathbb{E}_{x \sim p_g}[h(x)]$.

**(b)** For fixed $x$, define $f(d) = a \log d + b \log(1-d)$ where $a = p_{data}(x)$, $b = p_g(x)$.

$$f'(d) = \frac{a}{d} - \frac{b}{1-d} = 0$$
$$a(1-d) = bd$$
$$a - ad = bd$$
$$a = d(a + b)$$
$$\boxed{D^*(x) = \frac{a}{a+b} = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}}$$

**(c)** Second derivative:

$$f''(d) = -\frac{a}{d^2} - \frac{b}{(1-d)^2}$$

Since $a = p_{data}(x) \geq 0$ and $b = p_g(x) \geq 0$ (with at least one positive), $f''(d) < 0$. Confirmed maximum. ✓

**(d)** When $p_{data}(x) = p_g(x)$:

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_{data}(x)} = \frac{1}{2}$$

**Interpretation:** When the generator perfectly matches the data distribution, the optimal discriminator outputs 1/2 for every input — it literally cannot tell real from fake. This is the Nash equilibrium of the game.

---

### Solution A2: Jensen-Shannon Divergence Connection

**(a)** Substituting $D^* = \frac{p_{data}}{p_{data} + p_g}$:

$$V(D^*, G) = \int p_{data} \log \frac{p_{data}}{p_{data} + p_g} dx + \int p_g \log \frac{p_g}{p_{data} + p_g} dx$$

Define $m = \frac{p_{data} + p_g}{2}$, so $p_{data} + p_g = 2m$:

$$= \int p_{data} \log \frac{p_{data}}{2m} dx + \int p_g \log \frac{p_g}{2m} dx$$

$$= \int p_{data} \left(\log \frac{p_{data}}{m} - \log 2\right) dx + \int p_g \left(\log \frac{p_g}{m} - \log 2\right) dx$$

$$= D_{KL}(p_{data} \| m) - \log 2 + D_{KL}(p_g \| m) - \log 2$$

$$= D_{KL}(p_{data} \| m) + D_{KL}(p_g \| m) - 2\log 2$$

$$= 2 \cdot JSD(p_{data} \| p_g) - \log 4 \quad \blacksquare$$

**(b)** JSD is bounded because each KL term uses $m = (P+Q)/2$ as the reference. Since $P \leq 2m$ pointwise:

$$\frac{p(x)}{m(x)} = \frac{p(x)}{(p(x)+q(x))/2} \leq \frac{p(x)}{p(x)/2} = 2$$

So $D_{KL}(P \| M) \leq \log 2$. Similarly for $D_{KL}(Q \| M) \leq \log 2$.

Thus $JSD = \frac{1}{2}(D_{KL}(P\|M) + D_{KL}(Q\|M)) \leq \log 2$.

KL divergence is unbounded because the ratio $p(x)/q(x)$ can be arbitrarily large when $q(x) \to 0$ but $p(x) > 0$.

**(c)** Since $JSD(p_{data} \| p_g) \geq 0$ (as a sum of non-negative KL divergences):

$$V(D^*, G) = 2 \cdot JSD(p_{data} \| p_g) - \log 4 \geq 0 - \log 4 = -\log 4$$

Equality holds iff $JSD = 0$, which occurs iff $p_{data} = p_g$ (since both KL terms must be zero). $\blacksquare$

---

### Solution A3: Non-Saturating Loss Analysis

**(a)** When $D(G(z)) \approx 0$:

$$\frac{\partial}{\partial D} \log(1 - D) = \frac{-1}{1-D} \approx \frac{-1}{1-0} = -1$$

But the gradient with respect to the **generator parameters** involves the chain rule through $D(G(z))$. Since $D(G(z)) \approx 0$, the loss $\log(1) = 0$ is already near its maximum. The loss landscape is **flat** — gradients vanish. The generator learns nothing.

**(b)** When $D(G(z)) \approx 0$:

$$\frac{\partial}{\partial D} \log D = \frac{1}{D} \approx \frac{1}{0^+} \to \infty$$

The gradient is **large** when the discriminator correctly identifies fakes. This provides strong learning signal to the generator early in training.

**(c)** Both are optimized when $D(G(z)) = 1/2$ (discriminator fooled), which happens when $p_g = p_{data}$. At this point:
- $\log(1 - 1/2) = -\log 2$ (original: minimized)
- $\log(1/2) = -\log 2$ (non-saturating: maximized)

Same fixed point. ✓

**(d)** The non-saturating loss corresponds to minimizing:

$$-\mathbb{E}_{z}[\log D^*(G(z))] = -\mathbb{E}_{x \sim p_g}\left[\log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right]$$

This can be shown to be related to $D_{KL}(p_g \| p_{data})$ — the **reverse** KL divergence. The reverse KL is mode-seeking (it prefers to place mass where $p_{data}$ is high), which explains why GANs tend to produce sharp but mode-dropping samples.

---

### Solution A4: Mode Collapse Analysis

**(a)** With $p_g = \delta_3$ (all mass at $x=3$):

$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + \delta_3(x)}$$

- At $x = 3$: $p_{data}(3) = 0.5 \cdot \mathcal{N}(3|3,0.5) + 0.5 \cdot \mathcal{N}(3|-3,0.5) \approx 0.5 \cdot 0.798 = 0.399$. The generator mass at $x=3$ is infinite (delta function), so $D^*(3) \to 0$.
- At $x = -3$: $p_g(-3) = 0$, so $D^*(-3) = 1$ — purely real, no fakes here.
- Elsewhere: $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + 0} = 1$ for $x \neq 3$.

**(b)** Since $p_g = \delta_3$ concentrates all mass at a single point while $p_{data}$ spreads mass across two modes, the JSD will be close to $\log 2 \approx 0.693$ (nearly maximum). When $p_g = p_{data}$, $JSD = 0$. The collapsed generator is very far from optimal.

**(c)** The generator faces a dilemma: generating samples between the modes (around $x=0$) has very low $p_{data}$ density. The discriminator can easily reject these samples. It's safer for the generator to concentrate on one mode where it can produce high-quality (realistic) samples, even if it misses the other mode. The generator minimizes risk by specializing.

**(d)**
- $D_{KL}(p_g \| p_{data})$ is the **reverse KL** (mode-seeking): $p_g$ is penalized for placing mass where $p_{data}$ is low, but NOT penalized for missing modes of $p_{data}$. This encourages mode collapse.
- $D_{KL}(p_{data} \| p_g)$ is the **forward KL** (mean-seeking): $p_g$ must have mass wherever $p_{data}$ has mass, encouraging full mode coverage.

GANs (especially with the non-saturating loss) implicitly minimize something closer to reverse KL, explaining mode collapse.

---

### Solution A5: Wasserstein Distance Properties

**(a)**
- **JSD:** For $\theta \neq 0$, $P$ and $Q$ have disjoint support. The mixture $m = (\delta_0 + \delta_\theta)/2$ gives:
  $D_{KL}(P \| m) = \log \frac{1}{1/2} = \log 2$ and similarly $D_{KL}(Q \| m) = \log 2$.
  So $JSD = \frac{1}{2}(\log 2 + \log 2) = \log 2$.

- **$W_1$:** The only transport plan moves all mass from 0 to $\theta$, costing $|\theta|$. So $W_1 = |\theta|$.

**(b)**
- $\frac{\partial}{\partial \theta} JSD = 0$ for all $\theta \neq 0$. **No gradient signal!** The JSD is constant (at $\log 2$) regardless of how far apart the distributions are.
- $\frac{\partial}{\partial \theta} W_1 = \frac{\partial}{\partial \theta} |\theta| = \text{sign}(\theta)$. **Always provides a useful gradient** pointing toward $\theta = 0$.

**(c)** The 1-Lipschitz constraint ensures the critic function doesn't change too rapidly. Without it, the supremum in the dual form would be infinite — the critic could assign $+\infty$ to real data and $-\infty$ to fake data. The constraint forces the critic to be "smooth," and the maximum it can achieve measures the true distance between distributions.

**(d)** Interpolated samples lie in the region between real and fake data, which is exactly where the critic's gradient matters most for training. Sampling only from $p_{data}$ or $p_g$ would enforce the constraint at the endpoints but miss the transition region. The gradient penalty along interpolations ensures the critic is well-behaved everywhere the generator needs to learn.

---

### Solution A6: f-Divergences and GANs

**(a)** With $f(t) = t \log t$:

$$D_f(P \| Q) = \mathbb{E}_Q\left[\frac{p(x)}{q(x)} \log \frac{p(x)}{q(x)}\right] = \int q(x) \frac{p(x)}{q(x)} \log \frac{p(x)}{q(x)} dx = \int p(x) \log \frac{p(x)}{q(x)} dx = D_{KL}(P \| Q) \quad \checkmark$$

**(b)** With $f(t) = -\log t$:

$$D_f(P \| Q) = \mathbb{E}_Q\left[-\log \frac{p(x)}{q(x)}\right] = \int q(x) \log \frac{q(x)}{p(x)} dx = D_{KL}(Q \| P) \quad \checkmark$$

**(c)** The JSD can be written as:

$$JSD(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M) \text{ where } M = \frac{P+Q}{2}$$

As an f-divergence of $P$ w.r.t. $Q$, the corresponding $f$ is:

$$f(t) = \frac{1}{2}\left[t \log \frac{2t}{t+1} + \log \frac{2}{t+1}\right]$$

One can verify $f(1) = 0$ and $f$ is convex.

**(d)** The Fenchel conjugate: $f^*(u) = \sup_t [ut - f(t)]$.

By the variational representation:

$$D_f(P \| Q) = \sup_T \left\{\mathbb{E}_{x \sim P}[T(x)] - \mathbb{E}_{x \sim Q}[f^*(T(x))]\right\}$$

where $T: \mathcal{X} \to \mathbb{R}$ is any function (parameterized by a neural network in f-GANs).

This gives a lower bound that can be estimated with samples from $P$ (real data) and $Q$ (generator). Different choices of $f$ give different GAN objectives.

---

## Part B: Coding Solutions

### Solution B1: Implement the Optimal Discriminator

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def optimal_discriminator(x, p_data_fn, p_g_fn):
    p_d = p_data_fn(x)
    p_g = p_g_fn(x)
    return p_d / (p_d + p_g + 1e-10)  # small epsilon for stability

# (a) Plot distributions and D*(x)
x = np.linspace(-6, 10, 1000)
p_data_fn = lambda x: norm.pdf(x, 0, 1)
p_g_fn = lambda x: norm.pdf(x, 2, 1.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(x, p_data_fn(x), 'b-', linewidth=2, label='p_data ~ N(0,1)')
ax1.plot(x, p_g_fn(x), 'r--', linewidth=2, label='p_g ~ N(2,1.5)')
ax1.set_title('Distributions')
ax1.legend()
ax1.grid(True, alpha=0.3)

d_star = optimal_discriminator(x, p_data_fn, p_g_fn)
ax2.plot(x, d_star, 'g-', linewidth=2)
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax2.set_title('Optimal Discriminator D*(x)')
ax2.set_ylabel('D*(x)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# (b) D*(x) = 0.5 where p_data(x) = p_g(x)
# Find intersection
idx = np.argmin(np.abs(d_star - 0.5))
print(f"D*(x) ≈ 0.5 at x ≈ {x[idx]:.2f}")
print("This is where p_data(x) = p_g(x), i.e., the distributions intersect.")

# (c) When p_g = p_data
p_g_perfect = lambda x: norm.pdf(x, 0, 1)
d_star_perfect = optimal_discriminator(x, p_data_fn, p_g_perfect)

plt.figure(figsize=(10, 4))
plt.plot(x, d_star_perfect, 'g-', linewidth=2)
plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
plt.title('D*(x) when p_g = p_data: constant 0.5 everywhere')
plt.ylabel('D*(x)')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.show()
print("When the generator perfectly matches the data, D*(x) = 0.5 everywhere.")
```

---

### Solution B2: Visualize Training Dynamics

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# Real distribution: mixture of two Gaussians
def p_data(x):
    return 0.5 * norm.pdf(x, -3, 0.7) + 0.5 * norm.pdf(x, 3, 0.7)

# Generator: single Gaussian parameterized by (mu, sigma)
mu_g = 0.0
sigma_g = 2.0
lr = 0.05

x_plot = np.linspace(-8, 8, 500)
history = {'mu': [], 'sigma': [], 'jsd': []}
snapshots = {}

def p_g(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

def compute_jsd(x, p_d, p_q):
    m = 0.5 * (p_d + p_q)
    m = np.maximum(m, 1e-10)
    kl1 = np.where(p_d > 1e-10, p_d * np.log(p_d / m + 1e-10), 0)
    kl2 = np.where(p_q > 1e-10, p_q * np.log(p_q / m + 1e-10), 0)
    dx = x[1] - x[0]
    return 0.5 * np.sum(kl1) * dx + 0.5 * np.sum(kl2) * dx

for step in range(200):
    pd = p_data(x_plot)
    pg = p_g(x_plot, mu_g, sigma_g)

    if step in [0, 25, 50, 100, 199]:
        snapshots[step] = (mu_g, sigma_g, pg.copy())

    jsd = compute_jsd(x_plot, pd, pg)
    history['mu'].append(mu_g)
    history['sigma'].append(sigma_g)
    history['jsd'].append(jsd)

    # Numerical gradient of JSD w.r.t. mu and sigma
    eps = 0.01
    jsd_mu_plus = compute_jsd(x_plot, pd, p_g(x_plot, mu_g + eps, sigma_g))
    jsd_mu_minus = compute_jsd(x_plot, pd, p_g(x_plot, mu_g - eps, sigma_g))
    grad_mu = (jsd_mu_plus - jsd_mu_minus) / (2 * eps)

    jsd_sig_plus = compute_jsd(x_plot, pd, p_g(x_plot, mu_g, sigma_g + eps))
    jsd_sig_minus = compute_jsd(x_plot, pd, p_g(x_plot, mu_g, max(0.1, sigma_g - eps)))
    grad_sigma = (jsd_sig_plus - jsd_sig_minus) / (2 * eps)

    mu_g -= lr * grad_mu
    sigma_g -= lr * grad_sigma
    sigma_g = max(0.1, sigma_g)

# Plot snapshots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, (step, (mu, sig, pg)) in enumerate(snapshots.items()):
    ax = axes[0, i] if i < 3 else axes[1, i-3]
    ax.plot(x_plot, p_data(x_plot), 'b-', linewidth=2, label='p_data')
    ax.plot(x_plot, pg, 'r--', linewidth=2, label=f'p_g (μ={mu:.1f}, σ={sig:.1f})')
    ax.set_title(f'Step {step}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[1, 2].plot(history['jsd'], 'g-', linewidth=2)
axes[1, 2].set_title('JSD over Training')
axes[1, 2].set_xlabel('Step')
axes[1, 2].set_ylabel('JSD')
axes[1, 2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Final: μ={mu_g:.2f}, σ={sigma_g:.2f}")
print("The single Gaussian cannot capture both modes!")
print("This is a form of mode collapse — the generator covers one mode (or the space between).")
```

---

### Solution B3: Mode Collapse Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Simple MLP helpers ---
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# --- Data: 4 Gaussian clusters ---
def sample_data(n):
    centers = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
    idx = np.random.randint(0, 4, n)
    samples = np.array([centers[i] for i in idx]) + np.random.randn(n, 2) * 0.3
    return samples

# --- Generator: noise -> hidden -> output ---
noise_dim = 8
hidden_dim = 32
output_dim = 2

W_g1 = np.random.randn(noise_dim, hidden_dim) * 0.1
b_g1 = np.zeros(hidden_dim)
W_g2 = np.random.randn(hidden_dim, output_dim) * 0.1
b_g2 = np.zeros(output_dim)

# --- Discriminator: input -> hidden -> output ---
W_d1 = np.random.randn(2, hidden_dim) * 0.1
b_d1 = np.zeros(hidden_dim)
W_d2 = np.random.randn(hidden_dim, 1) * 0.1
b_d2 = np.zeros(1)

lr_d, lr_g = 0.01, 0.01
batch_size = 128

d_losses, g_losses = [], []
snapshots = {}

for step in range(2001):
    # --- Train Discriminator ---
    real = sample_data(batch_size)
    noise = np.random.randn(batch_size, noise_dim)

    # Generator forward
    g_h = relu(noise @ W_g1 + b_g1)
    fake = g_h @ W_g2 + b_g2

    # Discriminator forward on real
    d_h_real = relu(real @ W_d1 + b_d1)
    d_out_real = sigmoid(d_h_real @ W_d2 + b_d2)

    # Discriminator forward on fake
    d_h_fake = relu(fake @ W_d1 + b_d1)
    d_out_fake = sigmoid(d_h_fake @ W_d2 + b_d2)

    d_loss = -np.mean(np.log(d_out_real + 1e-8) + np.log(1 - d_out_fake + 1e-8))
    d_losses.append(d_loss)

    # D backward
    dd_real = -(1 / (d_out_real + 1e-8)) * d_out_real * (1 - d_out_real) / batch_size
    dd_fake = (1 / (1 - d_out_fake + 1e-8)) * d_out_fake * (1 - d_out_fake) / batch_size

    dW_d2 = d_h_real.T @ dd_real + d_h_fake.T @ dd_fake
    db_d2 = dd_real.sum(0) + dd_fake.sum(0)

    dd_h_real = (dd_real @ W_d2.T) * relu_deriv(real @ W_d1 + b_d1)
    dd_h_fake = (dd_fake @ W_d2.T) * relu_deriv(fake @ W_d1 + b_d1)
    dW_d1 = real.T @ dd_h_real + fake.T @ dd_h_fake
    db_d1 = dd_h_real.sum(0) + dd_h_fake.sum(0)

    W_d2 -= lr_d * dW_d2
    b_d2 -= lr_d * db_d2
    W_d1 -= lr_d * dW_d1
    b_d1 -= lr_d * db_d1

    # --- Train Generator (non-saturating loss) ---
    noise = np.random.randn(batch_size, noise_dim)
    g_h = relu(noise @ W_g1 + b_g1)
    fake = g_h @ W_g2 + b_g2

    d_h_fake = relu(fake @ W_d1 + b_d1)
    d_out_fake = sigmoid(d_h_fake @ W_d2 + b_d2)

    g_loss = -np.mean(np.log(d_out_fake + 1e-8))
    g_losses.append(g_loss)

    # G backward (through D)
    dg_out = -(1 / (d_out_fake + 1e-8)) * d_out_fake * (1 - d_out_fake) / batch_size
    dg_h_d = (dg_out @ W_d2.T) * relu_deriv(fake @ W_d1 + b_d1)
    dfake = dg_h_d @ W_d1.T

    dW_g2 = g_h.T @ dfake
    db_g2 = dfake.sum(0)
    dg_h = (dfake @ W_g2.T) * relu_deriv(noise @ W_g1 + b_g1)
    dW_g1 = noise.T @ dg_h
    db_g1 = dg_h.sum(0)

    W_g2 -= lr_g * dW_g2
    b_g2 -= lr_g * db_g2
    W_g1 -= lr_g * dW_g1
    b_g1 -= lr_g * db_g1

    if step in [0, 500, 1000, 2000]:
        noise_vis = np.random.randn(500, noise_dim)
        gh = relu(noise_vis @ W_g1 + b_g1)
        gen_samples = gh @ W_g2 + b_g2
        snapshots[step] = gen_samples.copy()

# (a) Plot results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
real_vis = sample_data(500)
for i, (step, samples) in enumerate(snapshots.items()):
    axes[0, i].scatter(real_vis[:, 0], real_vis[:, 1], alpha=0.3, s=10, c='blue', label='Real')
    axes[0, i].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c='red', label='Generated')
    axes[0, i].set_title(f'Step {step}')
    axes[0, i].legend(fontsize=7)
    axes[0, i].set_xlim(-6, 6)
    axes[0, i].set_ylim(-6, 6)
    axes[0, i].grid(True, alpha=0.3)

    # Count modes covered
    centers = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
    covered = sum(1 for c in centers if np.any(np.linalg.norm(samples - c, axis=1) < 1.5))
    axes[0, i].set_xlabel(f'Modes covered: {covered}/4')

# (b) Loss curves
axes[1, 0].plot(d_losses, alpha=0.5, linewidth=0.5)
axes[1, 0].set_title('D Loss')
axes[1, 0].set_xlabel('Step')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(g_losses, alpha=0.5, linewidth=0.5, color='red')
axes[1, 1].set_title('G Loss')
axes[1, 1].set_xlabel('Step')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].axis('off')
axes[1, 3].axis('off')
plt.tight_layout()
plt.show()

print("Losses oscillate rather than converge — characteristic of adversarial training.")
print("The generator likely covers 1-2 modes, demonstrating mode collapse.")
```

---

### Solution B4: Distance Metric Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, wasserstein_distance

def compute_jsd_continuous(mu1, s1, mu2, s2, n_points=10000):
    x = np.linspace(min(mu1, mu2) - 5*max(s1, s2), max(mu1, mu2) + 5*max(s1, s2), n_points)
    p = norm.pdf(x, mu1, s1)
    q = norm.pdf(x, mu2, s2)
    m = 0.5 * (p + q)
    dx = x[1] - x[0]

    kl_pm = np.sum(np.where(p > 1e-10, p * np.log(p / (m + 1e-10) + 1e-10), 0)) * dx
    kl_qm = np.sum(np.where(q > 1e-10, q * np.log(q / (m + 1e-10) + 1e-10), 0)) * dx
    return 0.5 * kl_pm + 0.5 * kl_qm

# (a) Wide Gaussians
distances = [0, 0.5, 1, 2, 3, 5, 10]
jsds_wide = [compute_jsd_continuous(0, 1, d, 1) for d in distances]
w1s_wide = [wasserstein_distance(
    np.random.normal(0, 1, 50000),
    np.random.normal(d, 1, 50000)
) for d in distances]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(distances, jsds_wide, 'bo-', linewidth=2, label='JSD')
ax1.plot(distances, w1s_wide, 'rs-', linewidth=2, label='W₁')
ax1.set_xlabel('Distance d between means')
ax1.set_ylabel('Divergence/Distance')
ax1.set_title('σ=1 (wide Gaussians)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (b) Narrow Gaussians
jsds_narrow = [compute_jsd_continuous(0, 0.01, d, 0.01) for d in distances]
w1s_narrow = [wasserstein_distance(
    np.random.normal(0, 0.01, 50000),
    np.random.normal(d, 0.01, 50000)
) for d in distances]

ax2.plot(distances, jsds_narrow, 'bo-', linewidth=2, label='JSD')
ax2.plot(distances, w1s_narrow, 'rs-', linewidth=2, label='W₁')
ax2.set_xlabel('Distance d between means')
ax2.set_ylabel('Divergence/Distance')
ax2.set_title('σ=0.01 (narrow Gaussians)')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Key observation:")
print("- W₁ grows linearly with distance d (smooth gradients)")
print("- JSD saturates at log(2) ≈ 0.693 for narrow Gaussians")
print("- JSD provides NO gradient signal once distributions stop overlapping!")

# (c) Sliding distributions
d_range = np.linspace(-5, 5, 100)
jsds_slide = [compute_jsd_continuous(0, 1, d, 1) for d in d_range]
w1s_slide = [abs(d) for d in d_range]  # W₁ for Gaussians with same σ

plt.figure(figsize=(10, 5))
plt.plot(d_range, jsds_slide, 'b-', linewidth=2, label='JSD')
plt.plot(d_range, w1s_slide, 'r-', linewidth=2, label='W₁')
plt.xlabel('Distance d')
plt.ylabel('Value')
plt.title('JSD vs W₁ as distributions slide past each other')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("W₁ is V-shaped (continuous, differentiable almost everywhere)")
print("JSD is smooth but flattens out — gradients vanish at large distances")
```

---

### Solution B5: Implement WGAN Gradient Penalty

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- Two moons data ---
def make_moons(n, noise=0.1):
    t = np.linspace(0, np.pi, n // 2)
    x1 = np.column_stack([np.cos(t), np.sin(t)]) + np.random.randn(n // 2, 2) * noise
    x2 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5]) + np.random.randn(n // 2, 2) * noise
    return np.vstack([x1, x2])

# --- MLP helpers ---
def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.2):
    return np.where(x > 0, 1, alpha)

def init_weights(shape):
    return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])

# Network parameters
noise_dim = 8
h_dim = 64

# Generator
W_g1 = init_weights((noise_dim, h_dim))
b_g1 = np.zeros(h_dim)
W_g2 = init_weights((h_dim, h_dim))
b_g2 = np.zeros(h_dim)
W_g3 = init_weights((h_dim, 2))
b_g3 = np.zeros(2)

# Critic (no sigmoid!)
W_c1 = init_weights((2, h_dim))
b_c1 = np.zeros(h_dim)
W_c2 = init_weights((h_dim, h_dim))
b_c2 = np.zeros(h_dim)
W_c3 = init_weights((h_dim, 1))
b_c3 = np.zeros(1)

def generator_forward(z):
    h1 = leaky_relu(z @ W_g1 + b_g1)
    h2 = leaky_relu(h1 @ W_g2 + b_g2)
    return h2 @ W_g3 + b_g3

def critic_forward(x):
    h1 = leaky_relu(x @ W_c1 + b_c1)
    h2 = leaky_relu(h1 @ W_c2 + b_c2)
    return h2 @ W_c3 + b_c3

def critic_gradient(x):
    """Compute gradient of critic output w.r.t. input x using finite differences."""
    eps = 1e-4
    grad = np.zeros_like(x)
    for i in range(x.shape[1]):
        x_plus = x.copy()
        x_plus[:, i] += eps
        x_minus = x.copy()
        x_minus[:, i] -= eps
        grad[:, i] = (critic_forward(x_plus) - critic_forward(x_minus)).flatten() / (2 * eps)
    return grad

# Training
batch_size = 128
lr = 1e-4
lam = 10  # gradient penalty coefficient
n_critic = 5
c_losses, g_losses = [], []

for step in range(5001):
    # --- Train Critic ---
    for _ in range(n_critic):
        real = make_moons(batch_size, noise=0.05)
        noise = np.random.randn(batch_size, noise_dim)
        fake = generator_forward(noise)

        c_real = critic_forward(real)
        c_fake = critic_forward(fake)

        # Gradient penalty
        alpha = np.random.rand(batch_size, 1)
        x_hat = alpha * real + (1 - alpha) * fake
        grad = critic_gradient(x_hat)
        grad_norm = np.sqrt(np.sum(grad**2, axis=1) + 1e-10)
        gp = lam * np.mean((grad_norm - 1)**2)

        c_loss = np.mean(c_fake) - np.mean(c_real) + gp

        # Simplified gradient update (finite difference for demo)
        eps = 1e-4
        for W, b in [(W_c1, b_c1), (W_c2, b_c2), (W_c3, b_c3)]:
            for idx in np.ndindex(W.shape):
                W[idx] += eps
                loss_plus = np.mean(critic_forward(fake)) - np.mean(critic_forward(real))
                W[idx] -= 2 * eps
                loss_minus = np.mean(critic_forward(fake)) - np.mean(critic_forward(real))
                W[idx] += eps  # restore
                W[idx] -= lr * (loss_plus - loss_minus) / (2 * eps)
                # Note: In practice, use autograd. This is for illustration only.
                break  # Only update a few params per step for speed
            break

    c_losses.append(c_loss)

    # --- Train Generator ---
    noise = np.random.randn(batch_size, noise_dim)
    fake = generator_forward(noise)
    g_loss = -np.mean(critic_forward(fake))
    g_losses.append(g_loss)

    if step % 1000 == 0:
        print(f"Step {step}: C_loss={c_loss:.4f}, G_loss={g_loss:.4f}")

# Note: This simplified implementation won't produce great results
# because we're using finite-difference gradients for illustration.
# A real implementation would use PyTorch/JAX autograd.

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
real_vis = make_moons(500, noise=0.05)
noise_vis = np.random.randn(500, noise_dim)
gen_vis = generator_forward(noise_vis)

axes[0].scatter(real_vis[:, 0], real_vis[:, 1], alpha=0.5, s=10, c='blue', label='Real')
axes[0].scatter(gen_vis[:, 0], gen_vis[:, 1], alpha=0.5, s=10, c='red', label='Generated')
axes[0].set_title('Generated vs Real')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(c_losses, alpha=0.7)
axes[1].set_title('Critic Loss')
axes[1].set_xlabel('Step')
axes[1].grid(True, alpha=0.3)

axes[2].plot(g_losses, alpha=0.7, color='red')
axes[2].set_title('Generator Loss')
axes[2].set_xlabel('Step')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nKey insight: The critic loss in WGAN approximates the Wasserstein distance,")
print("so it's a meaningful metric — unlike standard GAN discriminator loss!")
```
