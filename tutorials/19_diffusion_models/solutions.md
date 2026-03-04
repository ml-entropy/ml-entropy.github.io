# Tutorial 19: Solutions — Diffusion Models

---

## Part A: Theory Solutions

### Solution A1: Forward Process Derivation

**(a)** From $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$:

$$x_t = \sqrt{1-\beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

**(b)** Substitute recursively:

$x_1 = \sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1} \epsilon_1$ where $\alpha_t = 1-\beta_t$

$x_2 = \sqrt{\alpha_2} x_1 + \sqrt{1-\alpha_2} \epsilon_2 = \sqrt{\alpha_2 \alpha_1} x_0 + \sqrt{\alpha_2(1-\alpha_1)} \epsilon_1 + \sqrt{1-\alpha_2} \epsilon_2$

The noise terms: variance = $\alpha_2(1-\alpha_1) + (1-\alpha_2) = \alpha_2 - \alpha_1\alpha_2 + 1 - \alpha_2 = 1 - \alpha_1\alpha_2 = 1 - \bar{\alpha}_2$

So $x_2 = \sqrt{\bar{\alpha}_2} x_0 + \sqrt{1-\bar{\alpha}_2} \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$.

By induction: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$. $\blacksquare$

**(c)** $\bar{\alpha}_T = (1-0.01)^{1000} = 0.99^{1000} \approx e^{-10} \approx 4.5 \times 10^{-5}$

Signal remaining: $\sqrt{\bar{\alpha}_T} \approx 0.0067$ (essentially zero)
Noise: $\sqrt{1-\bar{\alpha}_T} \approx 0.99997$ (essentially 1)

After 1000 steps, the data is completely buried in noise.

**(d)**
```python
T = 1000
beta_min, beta_max = 1e-4, 0.02
betas = np.linspace(beta_min, beta_max, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)

plt.plot(np.sqrt(alpha_bar), label='Signal √ᾱₜ')
plt.plot(np.sqrt(1 - alpha_bar), label='Noise √(1-ᾱₜ)')
plt.xlabel('Timestep t')
plt.legend()
plt.title('Signal and Noise Strengths')
```

---

### Solution A2: Three Parameterizations

**(a)** From $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, solve for $x_0$:

$$x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon}{\sqrt{\bar{\alpha}_t}}$$

If we predict $\hat{\epsilon} = \epsilon_\theta(x_t, t)$, then our estimate of $x_0$ is:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}} \quad \blacksquare$$

**(b)** The conditional density is:

$$q(x_t | x_0) = \frac{1}{(2\pi(1-\bar{\alpha}_t))^{d/2}} \exp\left(-\frac{\|x_t - \sqrt{\bar{\alpha}_t}x_0\|^2}{2(1-\bar{\alpha}_t)}\right)$$

$$\log q(x_t | x_0) = -\frac{d}{2}\log(2\pi(1-\bar{\alpha}_t)) - \frac{\|x_t - \sqrt{\bar{\alpha}_t}x_0\|^2}{2(1-\bar{\alpha}_t)}$$

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{1-\bar{\alpha}_t}$$

Substituting $x_t - \sqrt{\bar{\alpha}_t}x_0 = \sqrt{1-\bar{\alpha}_t}\epsilon$:

$$= -\frac{\sqrt{1-\bar{\alpha}_t}\epsilon}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}} \quad \blacksquare$$

**(c)** $s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$

Equivalently: $\epsilon_\theta = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta$.

**(d)** Trade-offs:

- **$\epsilon$-prediction (DDPM)**: Best for training stability. The target $\epsilon$ has fixed variance $I$, so the loss has consistent scale across timesteps. Most practical implementations use this.
- **$x_0$-prediction**: Most intuitive — "what is the clean data?" Good for understanding and for tasks where you want to directly predict the output. Can be unstable at high noise levels.
- **Score prediction**: Best for theoretical understanding. Connects to score matching, Langevin dynamics, and SDEs. Natural for the probability flow ODE and flow matching connection.

---

### Solution A3: DDPM Loss from ELBO

**(a)**
$$L_{VLB} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{(-\log p_\theta(x_0|x_1))}_{L_0}$$

- $L_T$: Measures how close the final noised distribution is to the prior $\mathcal{N}(0,I)$. No parameters — just a constant.
- $L_{t-1}$: KL between the true denoising posterior (tractable given $x_0$) and our learned model. **This is the main training term.**
- $L_0$: Reconstruction likelihood — how well the final denoising step reconstructs the data.

**(b)** By Bayes' rule and the Markov property:

$$L_{t-1} = D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$$

Both are Gaussian. $q(x_{t-1}|x_t,x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$ and $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \tilde{\beta}_t I)$ (same variance).

**(c)** KL between Gaussians with identical covariance $\Sigma$:

$$D_{KL}(\mathcal{N}(\mu_1, \Sigma) \| \mathcal{N}(\mu_2, \Sigma)) = \frac{1}{2}(\mu_1 - \mu_2)^T \Sigma^{-1} (\mu_1 - \mu_2)$$

With $\Sigma = \tilde{\beta}_t I$:

$$L_{t-1} = \frac{1}{2\tilde{\beta}_t} \|\tilde{\mu}_t - \mu_\theta\|^2 \quad \blacksquare$$

**(d)** The true posterior mean is:

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)$$

If we parameterize $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$, then:

$$\|\tilde{\mu}_t - \mu_\theta\|^2 = \frac{\beta_t^2}{\alpha_t(1-\bar{\alpha}_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

So $L_{t-1} \propto \|\epsilon - \epsilon_\theta\|^2$ (up to a $t$-dependent constant). $\blacksquare$

**(e)** The full VLB weights each timestep by $\frac{\beta_t^2}{2\tilde{\beta}_t \alpha_t(1-\bar{\alpha}_t)}$. This down-weights high-noise timesteps where the model must do most of the heavy lifting. The simplified loss (equal weighting) forces the model to spend more capacity on these crucial high-noise steps, leading to better sample quality in practice.

---

### Solution A4: Score Function for Gaussians

**(a)** $p(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}$

$$s(x) = \frac{d}{dx}\log p(x) = -\frac{x - \mu}{\sigma^2}$$

The score points **toward the mean** — positive when $x < \mu$, negative when $x > \mu$. Magnitude is proportional to distance from the mean and inversely proportional to variance.

**(b)** $s(x) = \frac{\nabla_x p(x)}{p(x)} = \frac{\sum_k w_k \nabla_x \mathcal{N}(x; \mu_k, \sigma_k^2)}{\sum_k w_k \mathcal{N}(x; \mu_k, \sigma_k^2)}$

$$= \frac{\sum_k w_k \mathcal{N}(x; \mu_k, \sigma_k^2) \cdot \left(-\frac{x-\mu_k}{\sigma_k^2}\right)}{\sum_k w_k \mathcal{N}(x; \mu_k, \sigma_k^2)}$$

$$= \sum_k r_k(x) \cdot s_k(x)$$

where $r_k(x) = \frac{w_k \mathcal{N}(x; \mu_k, \sigma_k^2)}{\sum_j w_j \mathcal{N}(x; \mu_j, \sigma_j^2)}$ are the **responsibilities** (posterior mixture weights) and $s_k(x) = -(x-\mu_k)/\sigma_k^2$ are individual scores.

**(c)** For $p = 0.5\mathcal{N}(-3, 0.5) + 0.5\mathcal{N}(3, 0.5)$:
- At $x \ll -3$: score points right (toward $-3$, the nearest mode)
- At $x = -3$: score ≈ 0 (at the mode)
- At $x = 0$: score ≈ 0 (symmetric point, pulled equally left and right)
- At $x = 3$: score ≈ 0 (at the mode)
- At $x \gg 3$: score points left (toward $3$)

The score is zero at the modes and at the midpoint between them.

**(d)** Langevin dynamics has two parts:
1. **Gradient ascent**: $x + \eta s(x)$ moves toward higher probability (follows the score uphill)
2. **Noise injection**: $\sqrt{2\eta} z$ prevents the particle from getting stuck at a mode

Together, these implement a random walk that spends more time in high-probability regions. The stationary distribution of this process is exactly $p(x)$ — this is a fundamental result from stochastic calculus (the Fokker-Planck equation gives $p(x)$ as the equilibrium).

---

### Solution A5: Noise Schedule Analysis

**(a)** $\log \bar{\alpha}_t = \sum_{s=1}^t \log(1-\beta_s) \approx -\sum_{s=1}^t \beta_s$ (for small $\beta$)

For linear schedule: $\sum_{s=1}^t \beta_s \approx \int_0^t [\beta_{min} + (\beta_{max}-\beta_{min})s/T] ds = \beta_{min}t + \frac{(\beta_{max}-\beta_{min})t^2}{2T}$

So $\bar{\alpha}_t \approx \exp\left(-\beta_{min}t - \frac{(\beta_{max}-\beta_{min})t^2}{2T}\right)$

**(b)** $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$

$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = 1 - \frac{\cos^2(\frac{(t/T+s)\pi}{2(1+s)})}{\cos^2(\frac{((t-1)/T+s)\pi}{2(1+s)})}$

**(c)** $\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$

Linear schedule: SNR drops quickly in the middle and slowly at the extremes
Cosine schedule: SNR drops more uniformly across all timesteps

**(d)** With the linear schedule, the model doesn't benefit from early timesteps (almost no noise) or very late timesteps (almost pure noise). Most learning happens in a narrow band. The cosine schedule spreads the information destruction more evenly, so the model allocates capacity more uniformly across all noise levels, leading to better overall quality.

---

### Solution A6: DDIM Deterministic Sampling

**(a)** DDIM with $\sigma_t = 0$:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}\right)}_{\hat{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

**(b)** The update only uses $\bar{\alpha}_t$ and $\bar{\alpha}_{t-1}$, not the individual $\beta_t$. So we can use any subsequence $\bar{\alpha}_{\tau_1} > \bar{\alpha}_{\tau_2} > \cdots > \bar{\alpha}_{\tau_S}$ and apply the same formula, substituting $\bar{\alpha}_{\tau_{s-1}}$ for $\bar{\alpha}_{t-1}$.

**(c)** DDIM with $\sigma_t = 0$ is a deterministic function $x_T \mapsto x_0$. It's also invertible (we can run the same equations forward). This makes it a diffeomorphism from noise space to data space — exactly a normalizing flow! The log-likelihood can be computed via the change of variables formula.

**(d)**
- **DDPM**: Higher quality due to stochastic exploration. Best for maximum quality. Slow (T=1000 steps).
- **DDIM**: Deterministic = reproducible. Can skip steps for speed. Enables latent space interpolation and inversion. Best for applications needing consistency or speed.

---

## Part B: Coding Solutions

### Solution B1: Forward Process

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

T = 1000

# (a) Noise schedules
# Linear
beta_linear = np.linspace(1e-4, 0.02, T)
alpha_linear = 1 - beta_linear
alpha_bar_linear = np.cumprod(alpha_linear)

# Cosine
def cosine_schedule(T, s=0.008):
    t = np.arange(T + 1)
    f = np.cos((t/T + s) / (1+s) * np.pi/2)**2
    alpha_bar = f / f[0]
    alpha_bar = np.clip(alpha_bar, 1e-5, 1)
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 0, 0.999), alpha_bar[1:]

beta_cosine, alpha_bar_cosine = cosine_schedule(T)

# (b) Forward process
def forward_process(x0, t, alpha_bar):
    epsilon = np.random.randn(*x0.shape)
    sqrt_ab = np.sqrt(alpha_bar[t])
    sqrt_1mab = np.sqrt(1 - alpha_bar[t])
    if x0.ndim == 2:
        xt = sqrt_ab * x0 + sqrt_1mab * epsilon
    else:
        xt = sqrt_ab[:, None] * x0 + sqrt_1mab[:, None] * epsilon
    return xt, epsilon

# (c) Two moons data
def make_moons(n, noise=0.08):
    t = np.linspace(0, np.pi, n // 2)
    x1 = np.column_stack([np.cos(t), np.sin(t)])
    x2 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5])
    return np.vstack([x1, x2]) + np.random.randn(n, 2) * noise

data = make_moons(2000)

timesteps = [0, 100, 250, 500, 750, 999]
fig, axes = plt.subplots(2, 6, figsize=(20, 7))

for i, t in enumerate(timesteps):
    # Linear schedule
    xt_lin, _ = forward_process(data, t, alpha_bar_linear)
    axes[0, i].scatter(xt_lin[:, 0], xt_lin[:, 1], s=1, alpha=0.3, c='steelblue')
    snr = alpha_bar_linear[t] / (1 - alpha_bar_linear[t])
    axes[0, i].set_title(f'Linear t={t}\nSNR={snr:.2f}')
    axes[0, i].set_xlim(-4, 5)
    axes[0, i].set_ylim(-4, 4)
    axes[0, i].set_aspect('equal')
    axes[0, i].grid(True, alpha=0.3)

    # Cosine schedule
    xt_cos, _ = forward_process(data, t, alpha_bar_cosine)
    axes[1, i].scatter(xt_cos[:, 0], xt_cos[:, 1], s=1, alpha=0.3, c='coral')
    snr = alpha_bar_cosine[t] / (1 - alpha_bar_cosine[t])
    axes[1, i].set_title(f'Cosine t={t}\nSNR={snr:.2f}')
    axes[1, i].set_xlim(-4, 5)
    axes[1, i].set_ylim(-4, 4)
    axes[1, i].set_aspect('equal')
    axes[1, i].grid(True, alpha=0.3)

axes[0, 0].set_ylabel('Linear Schedule')
axes[1, 0].set_ylabel('Cosine Schedule')
plt.suptitle('Forward Diffusion Process', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# (d) Signal and noise strengths
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(np.sqrt(alpha_bar_linear), label='Linear', linewidth=2)
axes[0].plot(np.sqrt(alpha_bar_cosine), label='Cosine', linewidth=2)
axes[0].set_title('Signal Strength √ᾱₜ')
axes[0].set_xlabel('t')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(np.sqrt(1-alpha_bar_linear), label='Linear', linewidth=2)
axes[1].plot(np.sqrt(1-alpha_bar_cosine), label='Cosine', linewidth=2)
axes[1].set_title('Noise Strength √(1-ᾱₜ)')
axes[1].set_xlabel('t')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

snr_linear = alpha_bar_linear / (1 - alpha_bar_linear + 1e-10)
snr_cosine = alpha_bar_cosine / (1 - alpha_bar_cosine + 1e-10)
axes[2].semilogy(snr_linear, label='Linear', linewidth=2)
axes[2].semilogy(snr_cosine, label='Cosine', linewidth=2)
axes[2].set_title('Signal-to-Noise Ratio')
axes[2].set_xlabel('t')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Cosine schedule: more uniform SNR decrease → better use of all timesteps.")
print("Linear schedule: SNR drops sharply in the middle, wastes early/late steps.")
```

---

### Solution B2: Train DDPM on 2D Data

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

T = 200  # Fewer steps for 2D demo
beta = np.linspace(1e-4, 0.02, T)
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

data = make_moons(3000)

# Time embedding
def time_embedding(t, dim=16):
    """Sinusoidal time embedding."""
    freqs = np.exp(np.linspace(0, -4, dim // 2))
    args = t[:, None] * freqs[None, :]
    return np.column_stack([np.sin(args), np.cos(args)])

# Noise prediction network
class NoisePredictor:
    def __init__(self, t_dim=16, hidden=64):
        self.t_dim = t_dim
        inp_dim = 2 + t_dim
        self.W1 = np.random.randn(inp_dim, hidden) * 0.2
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * 0.2
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, 2) * 0.05
        self.b3 = np.zeros(2)

    def __call__(self, xt, t_idx):
        t_emb = time_embedding(t_idx.astype(float) / T, self.t_dim)
        inp = np.column_stack([xt, t_emb])
        h = np.tanh(inp @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

model = NoisePredictor(t_dim=16, hidden=48)

# Training
lr = 0.002
batch_size = 256
losses = []

for step in range(4000):
    idx = np.random.randint(0, len(data), batch_size)
    x0 = data[idx]
    t = np.random.randint(0, T, batch_size)
    epsilon = np.random.randn(batch_size, 2)

    # Forward process
    sqrt_ab = np.sqrt(alpha_bar[t])[:, None]
    sqrt_1mab = np.sqrt(1 - alpha_bar[t])[:, None]
    xt = sqrt_ab * x0 + sqrt_1mab * epsilon

    # Predict noise
    eps_pred = model(xt, t)
    loss = np.mean((eps_pred - epsilon)**2)
    losses.append(loss)

    # Update (stochastic numerical gradient)
    eps_g = 3e-4
    for param in model.params():
        flat = param.ravel()
        n_update = min(len(flat), 120)
        indices = np.random.choice(len(flat), n_update, replace=False)
        for j in indices:
            flat[j] += eps_g
            loss_p = np.mean((model(xt, t) - epsilon)**2)
            flat[j] -= 2*eps_g
            loss_m = np.mean((model(xt, t) - epsilon)**2)
            flat[j] += eps_g
            flat[j] -= lr * (loss_p - loss_m) / (2*eps_g)

    if step % 500 == 0:
        print(f"Step {step}: loss = {loss:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(losses, alpha=0.5)
plt.xlabel('Step')
plt.ylabel('MSE Loss')
plt.title('DDPM Training Loss')
plt.grid(True, alpha=0.3)
plt.show()

print("Training is stable and smooth — unlike GANs!")
```

---

### Solution B3: DDPM Sampling

```python
# Sampling
def ddpm_sample(model, n_samples, T, alpha, alpha_bar, beta):
    x = np.random.randn(n_samples, 2)
    trajectory = {T: x.copy()}

    for t in reversed(range(T)):
        t_batch = np.full(n_samples, t)
        eps_pred = model(x, t_batch)

        # Compute mean
        mu = (1 / np.sqrt(alpha[t])) * (
            x - (beta[t] / np.sqrt(1 - alpha_bar[t])) * eps_pred
        )

        # Add noise (except at t=0)
        if t > 0:
            sigma = np.sqrt(beta[t])
            z = np.random.randn(n_samples, 2)
            x = mu + sigma * z
        else:
            x = mu

        if t in [T-1, int(0.75*T), int(0.5*T), int(0.25*T), 0]:
            trajectory[t] = x.copy()

    return x, trajectory

samples, traj = ddpm_sample(model, 1000, T, alpha, alpha_bar, beta)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Generated vs Real
axes[0].scatter(data[:, 0], data[:, 1], s=2, alpha=0.2, c='steelblue', label='Real')
axes[0].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.3, c='coral', label='Generated')
axes[0].set_title('DDPM: Generated vs Real')
axes[0].legend()
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Denoising trajectory
keys = sorted(traj.keys(), reverse=True)[:6]
for i, t_key in enumerate(keys):
    axes[1].scatter(traj[t_key][:100, 0], traj[t_key][:100, 1],
                    s=5, alpha=0.3 + 0.1*i, label=f't={t_key}')
axes[1].set_title('Denoising Trajectory')
axes[1].legend(fontsize=8)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# Loss
axes[2].plot(losses, alpha=0.5)
axes[2].set_title('Training Loss')
axes[2].set_xlabel('Step')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("DDPM generates data by iteratively denoising pure noise.")
print(f"Used T={T} steps — more steps generally = better quality.")
```

---

### Solution B4: Langevin Dynamics

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# (a) Known distribution: mixture of 3 Gaussians
centers = np.array([[-2, 0], [2, 0], [0, 2.5]])
sigmas = [0.5, 0.5, 0.5]
weights = [1/3, 1/3, 1/3]

def log_prob(x):
    lps = []
    for c, s, w in zip(centers, sigmas, weights):
        lp = np.log(w) - 0.5 * np.sum((x - c)**2, axis=-1) / s**2 - np.log(2*np.pi*s**2)
        lps.append(lp)
    return np.log(np.sum(np.exp(np.array(lps) - np.max(lps, axis=0)), axis=0)) + np.max(lps, axis=0)

def score(x):
    """Exact score for Gaussian mixture."""
    probs = np.zeros(len(x))
    weighted_scores = np.zeros_like(x)
    for c, s, w in zip(centers, sigmas, weights):
        p = w * np.exp(-0.5 * np.sum((x - c)**2, axis=1) / s**2) / (2*np.pi*s**2)
        probs += p
        weighted_scores += p[:, None] * (-(x - c) / s**2)
    return weighted_scores / (probs[:, None] + 1e-10)

# Visualize score field
xx, yy = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-3, 5, 20))
grid = np.column_stack([xx.ravel(), yy.ravel()])
s_grid = score(grid)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].quiver(grid[:, 0], grid[:, 1], s_grid[:, 0], s_grid[:, 1],
               np.sqrt(s_grid[:, 0]**2 + s_grid[:, 1]**2), cmap='coolwarm', alpha=0.7)
axes[0].scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='x', zorder=5)
axes[0].set_title('Score Field ∇log p(x)')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# (b) Langevin dynamics
etas = [0.001, 0.01, 0.1]
colors = ['blue', 'green', 'orange']

for eta, color in zip(etas, colors):
    particles = np.random.randn(300, 2) * 3
    for step in range(2000):
        s = score(particles)
        noise = np.random.randn(*particles.shape)
        particles = particles + eta * s + np.sqrt(2*eta) * noise

    axes[1].scatter(particles[:, 0], particles[:, 1], s=3, alpha=0.3,
                    c=color, label=f'η={eta}')

axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='x', zorder=5)
axes[1].set_title('Langevin Dynamics Samples')
axes[1].set_aspect('equal')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Langevin dynamics follows the score (gradient of log-density) + noise.")
print("With proper step size, particles converge to samples from the target distribution.")
print("Too large η → unstable. Too small η → slow convergence.")
```
