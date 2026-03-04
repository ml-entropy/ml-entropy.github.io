# Tutorial 18: Solutions — Flow Matching

---

## Part A: Theory Solutions

### Solution A1: Continuity Equation

**(a)** For $p_t(x) = \mathcal{N}(x; \mu_t, \sigma_t^2)$:

$$p_t(x) = \frac{1}{\sigma_t \sqrt{2\pi}} \exp\left(-\frac{(x - \mu_t)^2}{2\sigma_t^2}\right)$$

$$\frac{\partial p_t}{\partial t} = p_t \left[\frac{(x - \mu_t) \dot{\mu}_t}{\sigma_t^2} - \frac{\dot{\sigma}_t}{\sigma_t} + \frac{(x-\mu_t)^2 \dot{\sigma}_t}{\sigma_t^3}\right]$$

where $\dot{\mu}_t = d\mu_t/dt = \mu_1$ and $\dot{\sigma}_t = d\sigma_t/dt = \sigma_1 - 1$.

**(b)** If $v(x) = c$ (constant), then:

$$\frac{\partial}{\partial x}(p_t v_t) = c \frac{\partial p_t}{\partial x}$$

And $p_t(x) = p_0(x - ct)$ gives:

$$\frac{\partial p_t}{\partial t} = -c \frac{\partial p_0}{\partial x}\bigg|_{x-ct} = -c \frac{\partial p_t}{\partial x}$$

So $\frac{\partial p_t}{\partial t} + c \frac{\partial p_t}{\partial x} = 0$. ✓ The continuity equation is satisfied.

**Interpretation:** A constant velocity field just shifts the distribution without changing its shape.

**(c)** With $v(x) = -x$: particles move toward the origin. The divergence is $\nabla \cdot v = \frac{\partial(-x)}{\partial x} = -1$.

From $\frac{d\log p_t}{dt} = -\nabla \cdot v = +1$, log-density **increases** everywhere at rate 1. Density increases at the origin because particles are being compressed inward. The Gaussian narrows over time: $\sigma_t = \sigma_0 e^{-t}$.

**(d)** The continuity equation says the total probability $\int p_t(x) dx = 1$ is preserved for all $t$. If it were violated, probability would be "created" or "destroyed" — like mass appearing or disappearing in a fluid. Physically, this would mean some particles vanish or spontaneously appear, which is non-physical for a valid probability flow.

---

### Solution A2: Conditional Velocity Field

**(a)** $x_t = (1-t)x_0 + tx_1$

$$\frac{dx_t}{dt} = -x_0 + x_1 = x_1 - x_0$$

This is **constant in time** — the velocity doesn't depend on $t$. The particle moves in a straight line at constant speed from $x_0$ to $x_1$.

**(b)** For $(x_0, x_1) = ((-2,0), (1,3))$:

$$x_t = (1-t)(-2, 0) + t(1, 3) = (-2 + 3t, 3t)$$

At $t=0$: $(-2, 0)$. At $t=0.5$: $(-0.5, 1.5)$. At $t=1$: $(1, 3)$.

The velocity is $(1-(-2), 3-0) = (3, 3)$ — constant throughout.

**(c)** Since $x_t$ is a linear function of $t$ (affine combination of two fixed points), the trajectory is a straight line in $\mathbb{R}^n$ parameterized by $t$.

This is related to optimal transport because the **optimal transport plan** between a point mass at $x_0$ and a point mass at $x_1$ moves mass along the straight line connecting them. The cost is $\|x_1 - x_0\|$, which is the Euclidean distance — the shortest possible path.

**(d)** Quadratic: $x_t = (1-t^2)x_0 + t^2 x_1$

$$\frac{dx_t}{dt} = -2t \cdot x_0 + 2t \cdot x_1 = 2t(x_1 - x_0)$$

The velocity is **time-dependent** (zero at $t=0$, maximum at $t=1$). The particle starts slow and accelerates. The path in space is still a straight line (same direction $x_1 - x_0$), but the **speed profile** is non-uniform. This makes the velocity field harder to learn and requires more ODE steps.

---

### Solution A3: Comparing Interpolation Schedules

**(a)** Conditional velocities:

1. **Linear:** $u_t = \frac{d}{dt}[(1-t)x_0 + tx_1] = x_1 - x_0$ (constant)
2. **VP:** $u_t = \frac{d}{dt}[\sqrt{1-t^2}x_0 + tx_1] = \frac{-t}{\sqrt{1-t^2}}x_0 + x_1$ (time-dependent, diverges as $t \to 1$)
3. **Cosine:** $u_t = \frac{d}{dt}[\cos(\frac{\pi t}{2})x_0 + \sin(\frac{\pi t}{2})x_1] = -\frac{\pi}{2}\sin(\frac{\pi t}{2})x_0 + \frac{\pi}{2}\cos(\frac{\pi t}{2})x_1$ (time-dependent)

**(b)** Expected squared norm (using $\mathbb{E}[x_0^T x_1] = 0$ since independent):

1. **Linear:** $\mathbb{E}[\|x_t\|^2] = (1-t)^2 d + t^2 d = d[(1-t)^2 + t^2] = d[1 - 2t + 2t^2]$
   Range: $d$ at $t=0$, $d/2$ at $t=0.5$, $d$ at $t=1$

2. **VP:** $\mathbb{E}[\|x_t\|^2] = (1-t^2)d + t^2 d = d$
   **Constant!** This is why it's called "variance preserving."

3. **Cosine:** $\mathbb{E}[\|x_t\|^2] = \cos^2(\frac{\pi t}{2})d + \sin^2(\frac{\pi t}{2})d = d$
   Also constant (since $\cos^2 + \sin^2 = 1$).

The VP and cosine paths keep variance most constant.

**(c)** Straightness $S = \|x_1 - x_0\|^2 / \int_0^1 \|dx_t/dt\|^2 dt$:

1. **Linear:** $\int_0^1 \|x_1 - x_0\|^2 dt = \|x_1-x_0\|^2$, so $S = 1$ (perfectly straight!)

2. **VP:** $\int_0^1 \|\frac{-t}{\sqrt{1-t^2}}x_0 + x_1\|^2 dt > \|x_1-x_0\|^2$ (the varying speed means $S < 1$)

3. **Cosine:** Similar — time-varying velocity means $S < 1$

**The linear path is the only one with $S = 1$.**

**(d)** The Euler method error per step is proportional to $(\Delta t)^2 \cdot \|d^2x/dt^2\|$ (the curvature). For a perfectly straight path, $d^2x/dt^2 = 0$, so Euler is **exact** regardless of step size! For curved paths (VP, cosine), smaller steps are needed to follow the curves accurately.

---

### Solution A4: CFM Equals FM

**(a)** FM loss: $\mathcal{L}_{FM} = \mathbb{E}_{t, x \sim p_t}[\|v_\theta(x,t) - u_t(x)\|^2]$

**(b)** CFM loss: $\mathcal{L}_{CFM} = \mathbb{E}_{t, x_0, x_1}[\|v_\theta(x_t, t) - u_t(x|x_1)\|^2]$

**(c)** Expand both:

$$\mathcal{L}_{FM} = \mathbb{E}_t \int p_t(x) \|v_\theta(x,t)\|^2 dx - 2\mathbb{E}_t \int p_t(x) v_\theta(x,t)^T u_t(x) dx + \text{const}$$

$$\mathcal{L}_{CFM} = \mathbb{E}_{t,x_0,x_1} \|v_\theta(x_t,t)\|^2 - 2\mathbb{E}_{t,x_0,x_1} v_\theta(x_t,t)^T u_t(x|x_1) + \text{const}$$

The first terms are equal: $\mathbb{E}_{x_0,x_1}[\|v_\theta(x_t,t)\|^2] = \int p_t(x)\|v_\theta(x,t)\|^2 dx$ (since $x_t \sim p_t$).

For the cross terms, by the tower property:

$$\mathbb{E}_{x_0,x_1}[v_\theta(x_t,t)^T u_t(x_t|x_1)] = \int p_t(x) v_\theta(x,t)^T \underbrace{\mathbb{E}[u_t(x|x_1)|x_t=x]}_{= u_t(x)} dx$$

Since the marginal velocity is $u_t(x) = \mathbb{E}[u_t(x|x_1)|x_t=x]$, the cross terms are also equal.

Therefore $\nabla_\theta \mathcal{L}_{CFM} = \nabla_\theta \mathcal{L}_{FM}$. $\blacksquare$

**(d)** This is crucial because:
- **FM is intractable**: computing $u_t(x)$ requires knowing $p_t(x)$ (the marginal density at time $t$), which requires integrating over all possible $(x_0, x_1)$ pairs
- **CFM is trivial**: $u_t(x|x_1) = x_1 - x_0$ is known analytically. No integration needed.
- Since their gradients match, optimizing the easy CFM objective is equivalent to optimizing the hard FM objective.

---

### Solution A5: Connection to Diffusion

**(a)** DDPM forward: $x_t = \sqrt{\bar{\alpha}_t} x_1 + \sqrt{1-\bar{\alpha}_t} \epsilon$

In flow matching notation with $x_0 = \epsilon$:
- $\alpha_t = \sqrt{\bar{\alpha}_t}$ (signal coefficient)
- $\sigma_t = \sqrt{1-\bar{\alpha}_t}$ (noise coefficient)

So $x_t = \alpha_t x_1 + \sigma_t x_0$.

**(b)** The DDPM loss predicts the noise: $\|\epsilon_\theta(x_t, t) - \epsilon\|^2 = \|\epsilon_\theta(x_t, t) - x_0\|^2$.

The CFM loss predicts velocity: $\|v_\theta(x_t, t) - u_t\|^2$ where $u_t = \dot{\alpha}_t x_1 + \dot{\sigma}_t x_0$.

Since $x_1 = (x_t - \sigma_t x_0) / \alpha_t$, we can express $u_t$ in terms of $x_0$:

$$u_t = \dot{\alpha}_t \frac{x_t - \sigma_t x_0}{\alpha_t} + \dot{\sigma}_t x_0 = \frac{\dot{\alpha}_t}{\alpha_t} x_t + \left(\dot{\sigma}_t - \frac{\dot{\alpha}_t \sigma_t}{\alpha_t}\right) x_0$$

The two losses differ by a linear reparameterization and a time-dependent weighting $\lambda(t)$ that depends on $\alpha_t, \sigma_t$. They have the same minimizer.

**(c)** Given $\epsilon_\theta$: $v_\theta(x, t) = \frac{\dot{\alpha}_t}{\alpha_t} x + \left(\dot{\sigma}_t - \frac{\dot{\alpha}_t \sigma_t}{\alpha_t}\right) \epsilon_\theta(x, t)$

**(d)** VP paths are curved (variance changes non-uniformly). OT paths are straight lines. When sampling with Euler, each step assumes constant velocity over $\Delta t$. If the path is curved, this assumption introduces error, requiring more steps. Straight paths = constant velocity = Euler is nearly exact = fewer steps needed. In practice: OT requires ~10 steps vs. VP requiring ~50-100 steps.

---

### Solution A6: Euler Method Analysis

**(a)** Euler method: $x_{k+1} = x_k + v_\theta(x_k, t_k) \cdot \Delta t$ where $t_k = k/N$ and $\Delta t = 1/N$.

Starting from $x_0 \sim \mathcal{N}(0, I)$, iterate $k = 0, 1, ..., N-1$ to get $x_N \approx x(1)$.

**(b)** For $v(x,t) = ax + b(t)$, the exact solution over $[t, t+\Delta t]$ involves $e^{a \Delta t} \approx 1 + a\Delta t + \frac{(a\Delta t)^2}{2} + ...$

Euler gives: $x(t + \Delta t) \approx x(t) + (ax(t) + b(t))\Delta t = x(t)(1 + a\Delta t) + b(t)\Delta t$

The error is: $x_{exact} - x_{Euler} = x(t) \frac{(a\Delta t)^2}{2} + O(\Delta t^3)$

Local error per step: $O(\Delta t^2)$. Global error after $N = 1/\Delta t$ steps: $O(\Delta t)$.

**(c)** If velocity is constant $v(x,t) = v_0$ (straight trajectories), then:

$x(t) = x_0 + v_0 \cdot t$, which is linear in $t$.

Euler with $N=1$: $x_1 = x_0 + v_0 \cdot 1 = x_0 + v_0$. This is **exact** because a linear function is perfectly captured by one Euler step. $\blacksquare$

**(d)** For OT paths (approximately straight): $N \approx 5-10$ steps should suffice. For VP paths (curved): $N \approx 50-100$ steps are typically needed. This 5-10x reduction in sampling steps is one of the main practical advantages of flow matching with OT paths.

---

## Part B: Coding Solutions

### Solution B1: Euler ODE Solver

```python
import numpy as np
import matplotlib.pyplot as plt

def euler_solve(v_fn, x0, t_start, t_end, n_steps):
    """Solve dx/dt = v(x, t) using Euler method."""
    dt = (t_end - t_start) / n_steps
    t = t_start
    x = x0.copy()
    trajectory = [x.copy()]
    times = [t]

    for _ in range(n_steps):
        x = x + v_fn(x, t) * dt
        t += dt
        trajectory.append(x.copy())
        times.append(t)

    return np.array(trajectory), np.array(times)

def rk4_solve(v_fn, x0, t_start, t_end, n_steps):
    """Solve dx/dt = v(x, t) using RK4 method."""
    dt = (t_end - t_start) / n_steps
    t = t_start
    x = x0.copy()
    trajectory = [x.copy()]

    for _ in range(n_steps):
        k1 = v_fn(x, t) * dt
        k2 = v_fn(x + k1/2, t + dt/2) * dt
        k3 = v_fn(x + k2/2, t + dt/2) * dt
        k4 = v_fn(x + k3, t + dt) * dt
        x = x + (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt
        trajectory.append(x.copy())

    return np.array(trajectory), None

# (b) Exponential decay: dx/dt = -x
v_decay = lambda x, t: -x
x0 = np.array([2.0])
t_exact = np.linspace(0, 3, 100)
x_exact = 2.0 * np.exp(-t_exact)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

step_counts = [5, 10, 50, 200]
errors = []
for N in step_counts:
    traj, times = euler_solve(v_decay, x0, 0, 3, N)
    axes[0].plot(times, traj[:, 0], 'o-', markersize=2, label=f'Euler N={N}')
    errors.append(np.abs(traj[-1, 0] - x_exact[-1]))

axes[0].plot(t_exact, x_exact, 'k-', linewidth=2, label='Exact')
axes[0].set_xlabel('t')
axes[0].set_ylabel('x(t)')
axes[0].set_title('Exponential Decay: dx/dt = -x')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].loglog(step_counts, errors, 'ro-', linewidth=2, label='Euler error')
axes[1].set_xlabel('Number of steps N')
axes[1].set_ylabel('|x(3) - x_exact(3)|')
axes[1].set_title('Error vs Step Count (log-log)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# (c) Circular ODE
v_circle = lambda x, t: np.array([-x[1], x[0]])
x0_circ = np.array([1.0, 0.0])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
theta = np.linspace(0, 2*np.pi, 100)

for N in [10, 50, 200]:
    traj, _ = euler_solve(v_circle, x0_circ, 0, 2*np.pi, N)
    axes[0].plot(traj[:, 0], traj[:, 1], '-', label=f'Euler N={N}')

axes[0].plot(np.cos(theta), np.sin(theta), 'k--', linewidth=2, label='Exact circle')
axes[0].set_aspect('equal')
axes[0].set_title('Circular ODE (Euler spirals outward!)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# (d) RK4 comparison
for N in [10, 50]:
    traj_euler, _ = euler_solve(v_circle, x0_circ, 0, 2*np.pi, N)
    traj_rk4, _ = rk4_solve(v_circle, x0_circ, 0, 2*np.pi, N)
    axes[1].plot(traj_euler[:, 0], traj_euler[:, 1], '--', alpha=0.5, label=f'Euler N={N}')
    axes[1].plot(traj_rk4[:, 0], traj_rk4[:, 1], '-', label=f'RK4 N={N}')

axes[1].plot(np.cos(theta), np.sin(theta), 'k--', linewidth=1, label='Exact')
axes[1].set_aspect('equal')
axes[1].set_title('Euler vs RK4')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Euler with N=200 ≈ RK4 with N=10 for this problem.")
print("RK4 is ~20x more efficient per evaluation for curved trajectories.")
```

---

### Solution B2: Train Flow Matching on 2D Data

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# (a) Two moons data
def make_moons(n, noise=0.08):
    t = np.linspace(0, np.pi, n // 2)
    x1 = np.column_stack([np.cos(t), np.sin(t)])
    x2 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5])
    return np.vstack([x1, x2]) + np.random.randn(n, 2) * noise

data = make_moons(2000)

# (b) Velocity network: MLP [x1, x2, t] -> [v1, v2]
class VelocityNetwork:
    def __init__(self, hidden=64):
        self.W1 = np.random.randn(3, hidden) * 0.3
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * 0.3
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, 2) * 0.1
        self.b3 = np.zeros(2)

    def forward(self, x, t):
        """x: (batch, 2), t: (batch, 1) -> (batch, 2)"""
        inp = np.column_stack([x, t])
        h = np.tanh(inp @ self.W1 + self.b1)
        h = np.tanh(h @ self.W2 + self.b2)
        return h @ self.W3 + self.b3

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

net = VelocityNetwork(hidden=64)

# (c) Training loop
lr = 0.001
batch_size = 256
losses = []

for step in range(3000):
    # Sample data, noise, time
    idx = np.random.randint(0, len(data), batch_size)
    x1 = data[idx]
    x0 = np.random.randn(batch_size, 2)
    t = np.random.rand(batch_size, 1)

    # Interpolate
    xt = (1 - t) * x0 + t * x1

    # Target velocity
    target = x1 - x0

    # Predict
    pred = net.forward(xt, t)

    # Loss
    loss = np.mean((pred - target)**2)
    losses.append(loss)

    # Numerical gradient update
    eps = 5e-4
    for param in net.get_params():
        grad = np.zeros_like(param)
        flat = param.ravel()
        for j in range(min(len(flat), 200)):  # Update subset for speed
            jj = np.random.randint(0, len(flat))
            flat[jj] += eps
            pred_p = net.forward(xt, t)
            loss_p = np.mean((pred_p - target)**2)
            flat[jj] -= 2*eps
            pred_m = net.forward(xt, t)
            loss_m = np.mean((pred_m - target)**2)
            flat[jj] += eps
            flat[jj] -= lr * (loss_p - loss_m) / (2*eps)

    if step % 500 == 0:
        print(f"Step {step}: loss = {loss:.4f}")

# (d) Generate samples
N_steps = 20
x_gen = np.random.randn(1000, 2)
trajectories = [x_gen.copy()]

for k in range(N_steps):
    t_k = np.full((len(x_gen), 1), k / N_steps)
    v = net.forward(x_gen, t_k)
    x_gen = x_gen + v / N_steps
    trajectories.append(x_gen.copy())

# (e) Visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Real vs Generated
axes[0, 0].scatter(data[:, 0], data[:, 1], s=3, alpha=0.3, c='steelblue', label='Real')
axes[0, 0].scatter(x_gen[:, 0], x_gen[:, 1], s=3, alpha=0.3, c='coral', label='Generated')
axes[0, 0].set_title('Real vs Generated')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Training loss
axes[0, 1].plot(losses, alpha=0.7)
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('MSE Loss')
axes[0, 1].set_title('Training Loss')
axes[0, 1].grid(True, alpha=0.3)

# Trajectories
n_show = 30
for i in range(n_show):
    path = np.array([traj[i] for traj in trajectories])
    axes[0, 2].plot(path[:, 0], path[:, 1], 'b-', alpha=0.2, linewidth=0.5)
    axes[0, 2].scatter(path[0, 0], path[0, 1], s=10, c='gray', alpha=0.5)
    axes[0, 2].scatter(path[-1, 0], path[-1, 1], s=10, c='coral', alpha=0.5)
axes[0, 2].set_title('Sample Trajectories (noise → data)')
axes[0, 2].grid(True, alpha=0.3)

# Vector fields at different times
for i, t_val in enumerate([0.0, 0.5, 1.0]):
    ax = axes[1, i]
    xx, yy = np.meshgrid(np.linspace(-3, 4, 20), np.linspace(-2, 3, 20))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    t_grid = np.full((len(grid_pts), 1), t_val)
    v_grid = net.forward(grid_pts, t_grid)

    ax.quiver(grid_pts[:, 0], grid_pts[:, 1], v_grid[:, 0], v_grid[:, 1],
              np.sqrt(v_grid[:, 0]**2 + v_grid[:, 1]**2),
              cmap='coolwarm', alpha=0.7)
    ax.scatter(data[:200, 0], data[:200, 1], s=1, alpha=0.2, c='steelblue')
    ax.set_title(f'Vector field at t={t_val}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Flow Matching: Complete Results', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("🔑 Flow matching training is just regression — predict velocity = (x1 - x0)!")
print("Sampling solves the ODE with Euler method in ~20 steps.")
```

---

### Solution B3: Compare Sampling Step Counts

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# Using trained model from B2
step_counts = [1, 2, 5, 10, 20, 50, 100]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
qualities = []

for i, N in enumerate(step_counts):
    x_samples = np.random.randn(1000, 2)
    for k in range(N):
        t_k = np.full((len(x_samples), 1), k / N)
        v = net.forward(x_samples, t_k)
        x_samples = x_samples + v / N

    row, col = i // 4, i % 4
    axes[row, col].scatter(data[:500, 0], data[:500, 1], s=2, alpha=0.2, c='steelblue')
    axes[row, col].scatter(x_samples[:, 0], x_samples[:, 1], s=2, alpha=0.3, c='coral')
    axes[row, col].set_title(f'N={N} steps')
    axes[row, col].set_xlim(-2, 3)
    axes[row, col].set_ylim(-1.5, 2)
    axes[row, col].grid(True, alpha=0.3)

    # Quality metric: Wasserstein distance on each dimension
    w1 = wasserstein_distance(data[:, 0], x_samples[:, 0])
    w2 = wasserstein_distance(data[:, 1], x_samples[:, 1])
    qualities.append(w1 + w2)

axes[1, 3].semilogy(step_counts, qualities, 'ro-', linewidth=2, markersize=8)
axes[1, 3].set_xlabel('Number of Euler Steps')
axes[1, 3].set_ylabel('Wasserstein Distance (lower = better)')
axes[1, 3].set_title('Sample Quality vs Steps')
axes[1, 3].grid(True, alpha=0.3)

plt.suptitle('Effect of Number of ODE Steps on Sample Quality', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("With OT paths, N=10-20 steps is usually sufficient.")
print("Diffusion models typically need N=50-1000 steps for similar quality.")
print("This is a 5-50x speedup at inference time!")
```
