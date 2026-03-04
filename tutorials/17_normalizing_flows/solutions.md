# Tutorial 17: Solutions — Normalizing Flows

---

## Part A: Theory Solutions

### Solution A1: 1D Change of Variables

**(a)** $f(x) = 3x - 2$, so $f^{-1}(y) = (y+2)/3$ and $|df^{-1}/dy| = 1/3$.

$$p_Y(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{((y+2)/3)^2}{2}\right) \cdot \frac{1}{3} = \frac{1}{3\sqrt{2\pi}} \exp\left(-\frac{(y+2)^2}{18}\right)$$

This is $\mathcal{N}(-2, 9)$: mean $= 3 \cdot 0 - 2 = -2$, variance $= 3^2 \cdot 1 = 9$. ✓

**(b)** $f(x) = x^3$, so $f^{-1}(y) = y^{1/3}$ and $|df^{-1}/dy| = \frac{1}{3}|y|^{-2/3}$.

$$p_Y(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^{2/3}}{2}\right) \cdot \frac{1}{3|y|^{2/3}}$$

This is not a standard named distribution. It's a "stretched Gaussian" that's heavier-tailed than the original.

**(c)** $f(x) = e^x$, so $f^{-1}(y) = \ln(y)$ for $y > 0$, and $|df^{-1}/dy| = 1/y$.

$$p_Y(y) = \frac{1}{y\sqrt{2\pi}} \exp\left(-\frac{(\ln y)^2}{2}\right), \quad y > 0$$

This is the **log-normal distribution**. To verify $\int_0^\infty p_Y(y) dy = 1$: substitute $u = \ln y$, $du = dy/y$:
$$\int_0^\infty \frac{1}{y\sqrt{2\pi}} e^{-(\ln y)^2/2} dy = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-u^2/2} du = 1 \quad \checkmark$$

**(d)** $f(x) = \sigma(x) = 1/(1+e^{-x})$, so $f^{-1}(y) = \ln(y/(1-y))$ (logit function).

$$\frac{df^{-1}}{dy} = \frac{1}{y(1-y)}$$

$$p_Y(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{[\ln(y/(1-y))]^2}{2}\right) \cdot \frac{1}{y(1-y)}, \quad y \in (0,1)$$

This is related to the **logit-normal distribution**, used in compositional data analysis.

---

### Solution A2: Jacobian Computation

**(a)** For $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$: the Jacobian is simply $J = A$.

$$J = A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}, \quad \det(J) = 2 \cdot 3 - 1 \cdot 0 = 6$$

A unit circle gets mapped to an ellipse with area scaled by $|\det(A)| = 6$.

**(b)** Rotation matrix:

$$J = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

$$\det(J) = \cos^2\theta + \sin^2\theta = 1$$

Rotations preserve areas (and therefore densities) because they are orthogonal transformations. $|\det(J)| = 1$ everywhere means no stretching or compression.

**(c)** Shear: $f(x_1, x_2) = (x_1, x_2 + x_1^2)$

$$J = \begin{pmatrix} 1 & 0 \\ 2x_1 & 1 \end{pmatrix}, \quad \det(J) = 1$$

This is volume-preserving! Shear transformations distort shapes but don't change areas, so probability densities remain unchanged.

**(d)** $f(x_1, x_2) = (x_1 e^{x_2}, x_2)$

$$J = \begin{pmatrix} e^{x_2} & x_1 e^{x_2} \\ 0 & 1 \end{pmatrix}, \quad \det(J) = e^{x_2}$$

- When $x_2 > 0$: $\det(J) > 1$, the transformation stretches space → density decreases
- When $x_2 < 0$: $\det(J) < 1$, the transformation compresses → density increases
- When $x_2 = 0$: $\det(J) = 1$, volume-preserving

---

### Solution A3: Coupling Layer Properties

**(a)** The Jacobian in block form:

$$J = \frac{\partial(y_1, y_2)}{\partial(x_1, x_2)} = \begin{pmatrix} I & 0 \\ \frac{\partial y_2}{\partial x_1} & \text{diag}(\exp(s(x_1))) \end{pmatrix}$$

The upper-right block is zero because $y_1 = x_1$ doesn't depend on $x_2$.

**(b)** For a lower-triangular block matrix: $\det(J) = \det(I) \cdot \det(\text{diag}(\exp(s(x_1))))$

$$\det(J) = 1 \cdot \prod_{i} \exp(s_i(x_1)) = \exp\left(\sum_i s_i(x_1)\right)$$

This is $O(d/2)$ to compute — just sum the output of $s$! Compare with $O(d^3)$ for a general Jacobian.

**(c)** Inverse:
$$x_1 = y_1 \quad \text{(direct copy)}$$
$$x_2 = (y_2 - t(y_1)) \odot \exp(-s(y_1))$$

Since $y_1 = x_1$, we can use $y_1$ directly in $s$ and $t$. Only **one** evaluation of $s$ and $t$ is needed for the inverse — same cost as the forward pass!

**(d)** Layer 1 (fix $x_1$): $y_1 = x_1$, $y_2 = x_2 \cdot \exp(s_1(x_1)) + t_1(x_1)$

Layer 2 (fix $y_2$): $z_1 = y_1 \cdot \exp(s_2(y_2)) + t_2(y_2)$, $z_2 = y_2$

Composite: ALL dimensions are transformed. The composite Jacobian is the product of two triangular matrices — it is NOT triangular in general, but its determinant is simply the product of the individual determinants.

---

### Solution A4: Log-Likelihood for Composed Flows

**(a)**
$$\log p(x) = \log p_0(z_0) - \log|\det(J_{f_1})| - \log|\det(J_{f_2})| - \log|\det(J_{f_3})|$$

where $z_0 = f_1^{-1}(f_2^{-1}(f_3^{-1}(x)))$.

**(b)** Steps for $x = (1, 2)$:
1. Compute $z_2 = f_3^{-1}(x)$ and $\log|\det(J_{f_3}^{-1})|$ (or equivalently $-\log|\det(J_{f_3})|$)
2. Compute $z_1 = f_2^{-1}(z_2)$ and $\log|\det(J_{f_2}^{-1})|$
3. Compute $z_0 = f_1^{-1}(z_1) = A^{-1}(z_1 - b)$ and $\log|\det(J_{f_1}^{-1})| = -\log|\det(A)| = -\log 2$
4. Compute $\log p_0(z_0) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\|z_0\|^2$
5. Sum all terms

**(c)** Must store: all intermediate activations $z_0, z_1, z_2$ (for backward pass through each layer), plus the Jacobian determinant at each layer.

**(d)** Discrete flow: $O(K)$ memory (store $K$ intermediate states). Continuous flow with adjoint: $O(1)$ memory (only store final state, recompute intermediates during backward pass). However, the adjoint method requires solving an ODE backward, which can be slow.

---

### Solution A5: Flow Topology

**(a)** A diffeomorphism is a continuous, continuously differentiable bijection with a continuous inverse. By the intermediate value theorem (in 1D) or by topological properties of continuous maps (in nD), continuous maps preserve connectedness. Therefore, a diffeomorphism cannot map a connected set to a disconnected set.

**(b)** Flows handle multi-modal distributions by creating "channels" of probability: the Gaussian mass is routed through thin, high-density corridors to different modes. With enough layers, the flow can create arbitrarily thin bridges between modes. In practice, this requires many coupling layers — typically 8-32 for moderately complex distributions.

**(c)** Let $f$ and $g$ be diffeomorphisms. Then:
- $g \circ f$ is bijective (composition of bijections)
- $g \circ f$ is $C^1$ (composition of $C^1$ maps)
- $(g \circ f)^{-1} = f^{-1} \circ g^{-1}$ is $C^1$ (composition of $C^1$ maps)
Therefore $g \circ f$ is a diffeomorphism. $\blacksquare$

**(d)** The flow needs to split the Gaussian into two lobes:
- In the region between the two modes (around $x_1 = 0$), the flow must create a "valley" — compressing probability → $|\det(J)| < 1$ (density increases in latent space, meaning this data space region gets less probability)
- At the two mode locations ($x_1 \approx \pm 3$), the flow stretches the Gaussian → $|\det(J)| > 1$ to create the two bumps
- The flow essentially creates an hourglass shape: stretching at the modes, compressing in between

---

### Solution A6: Autoregressive vs Coupling Flows

**(a)** MAF for $n=3$:
$$y_1 = x_1 \cdot e^{\alpha_1} + \mu_1 \quad \text{(no conditioning)}$$
$$y_2 = x_2 \cdot e^{\alpha_2(x_1)} + \mu_2(x_1)$$
$$y_3 = x_3 \cdot e^{\alpha_3(x_1, x_2)} + \mu_3(x_1, x_2)$$

Jacobian:
$$J = \begin{pmatrix} e^{\alpha_1} & 0 & 0 \\ \frac{\partial y_2}{\partial x_1} & e^{\alpha_2(x_1)} & 0 \\ \frac{\partial y_3}{\partial x_1} & \frac{\partial y_3}{\partial x_2} & e^{\alpha_3(x_1,x_2)} \end{pmatrix}$$

Lower-triangular! $\det(J) = e^{\alpha_1} \cdot e^{\alpha_2(x_1)} \cdot e^{\alpha_3(x_1,x_2)} = \exp(\alpha_1 + \alpha_2 + \alpha_3)$

**(b)** IAF: $y_i = x_i \cdot e^{\alpha_i(y_1,...,y_{i-1})} + \mu_i(y_1,...,y_{i-1})$

- **Sampling is fast:** Given $x \sim \mathcal{N}(0,I)$, compute $y_1$ first, then $y_2$ using $y_1$, etc. Each step uses only previously computed $y$'s → sequential but fast.
- **Density evaluation is slow:** Given $y$, to find $x$ we need $x_i$ which depends on $\alpha_i(y_1,...,y_{i-1})$. Computing all $\alpha_i$ requires a separate network pass for each dimension → slow.

MAF is the opposite: density evaluation only needs one forward pass, but sampling is sequential.

**(c)** A coupling layer sets $\alpha_i$ and $\mu_i$ to depend only on $x_{1:d/2}$ for $i > d/2$, and to be identity for $i \leq d/2$. This is a special case where the autoregressive dependency structure has a block form.

**(d)** Coupling layers allow BOTH forward and inverse passes to be computed in **parallel** (not sequentially). MAF requires sequential computation for sampling; IAF requires sequential computation for density. Coupling layers sacrifice some expressiveness for this parallelism, making them the practical choice when both operations are needed.

---

## Part B: Coding Solutions

### Solution B1: 1D Normalizing Flow

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)
x_range = np.linspace(-4, 4, 1000)

# (a) Change of variables implementation
def transform_density(y_range, f_inv, f_inv_deriv, base_pdf):
    """Compute transformed density using change of variables."""
    x = f_inv(y_range)
    jacobian = np.abs(f_inv_deriv(y_range))
    return base_pdf(x) * jacobian

# (b) f(x) = sinh(x)
f = np.sinh
f_inv = np.arcsinh  # inverse of sinh
f_inv_deriv = lambda y: 1 / np.sqrt(y**2 + 1)

# Sample and transform
samples_x = np.random.randn(10000)
samples_y = f(samples_x)

y_range = np.linspace(-10, 10, 1000)
p_y = transform_density(y_range, f_inv, f_inv_deriv, norm.pdf)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(x_range, norm.pdf(x_range), 'b-', lw=2)
axes[0].set_title('Original: N(0,1)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(y_range, p_y, 'r-', lw=2, label='Theory')
axes[1].hist(samples_y, bins=80, density=True, alpha=0.3, label='Samples')
axes[1].set_title('Transformed: Y = sinh(X)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# (c) f(x) = x + 0.5*sin(2x)
f2 = lambda x: x + 0.5 * np.sin(2*x)
# Numerical inverse via bisection
def f2_inv(y, tol=1e-8):
    result = np.zeros_like(y)
    for i, yi in enumerate(y):
        lo, hi = yi - 3, yi + 3
        for _ in range(100):
            mid = (lo + hi) / 2
            if f2(mid) < yi:
                lo = mid
            else:
                hi = mid
        result[i] = (lo + hi) / 2
    return result

f2_deriv = lambda x: 1 + np.cos(2*x)  # f'(x)
# |df_inv/dy| = 1/|f'(f_inv(y))|

samples_y2 = f2(samples_x)
y2_range = np.linspace(-5, 5, 500)
x2_inv = f2_inv(y2_range)
p_y2 = norm.pdf(x2_inv) / np.abs(f2_deriv(x2_inv))

axes[2].plot(y2_range, p_y2, 'r-', lw=2, label='Theory')
axes[2].hist(samples_y2, bins=80, density=True, alpha=0.3, label='Samples')
axes[2].set_title('Y = X + 0.5·sin(2X)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("f(x) = x + 0.5*sin(2x) is invertible because f'(x) = 1 + cos(2x) ≥ 0")
print("(it equals 0 only at isolated points, so f is still strictly increasing)")

# (d) Composed transformations
f_layers = [
    lambda x: x + 0.3 * np.sin(x),
    lambda x: 1.5 * x,
    lambda x: x + 0.5 * np.tanh(2*x),
]

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
z = np.random.randn(20000)
axes[0].hist(z, bins=80, density=True, alpha=0.5, color='steelblue')
axes[0].set_title('z₀ ~ N(0,1)')
axes[0].grid(True, alpha=0.3)

for i, f_layer in enumerate(f_layers):
    z = f_layer(z)
    axes[i+1].hist(z, bins=80, density=True, alpha=0.5, color='coral')
    axes[i+1].set_title(f'After f_{i+1}')
    axes[i+1].grid(True, alpha=0.3)

plt.suptitle('Composing Simple Transforms: N(0,1) → Complex Distribution', y=1.02)
plt.tight_layout()
plt.show()
```

---

### Solution B2: Visualizing 2D Transformations

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def make_grid(n=20, lim=3):
    """Create a grid of points."""
    t = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(t, t)
    return np.column_stack([xx.ravel(), yy.ravel()])

def plot_grid_transform(ax, points_before, points_after, n=20, title=''):
    """Plot grid deformation."""
    for i in range(n):
        # Horizontal lines
        idx = np.arange(i*n, (i+1)*n)
        ax.plot(points_after[idx, 0], points_after[idx, 1], 'b-', alpha=0.3, lw=0.5)
        # Vertical lines
        idx = np.arange(i, n*n, n)
        ax.plot(points_after[idx, 0], points_after[idx, 1], 'r-', alpha=0.3, lw=0.5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

n = 25
grid = make_grid(n, 3)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# (a) Affine transformation
A = np.array([[2, 1], [0, 3]])
grid_affine = grid @ A.T
plot_grid_transform(axes[0, 0], grid, grid_affine, n, f'Affine (det={np.linalg.det(A):.0f})')

# (b) Planar flow
u = np.array([0.5, 1.0])
w = np.array([1.0, 0.5])
b = 0.0

def planar_flow(z):
    return z + np.outer(np.tanh(z @ w + b), u)

grid_planar = planar_flow(grid)
plot_grid_transform(axes[0, 1], grid, grid_planar, n, 'Planar Flow')

# Jacobian det for planar flow
def planar_det(z):
    h_prime = 1 - np.tanh(z @ w + b)**2  # derivative of tanh
    return np.abs(1 + h_prime * (u @ w))

# (c) Coupling layer
def s_fn(x1): return 0.5 * np.sin(x1)
def t_fn(x1): return np.cos(x1)

def coupling_layer(z):
    y = z.copy()
    y[:, 1] = z[:, 1] * np.exp(s_fn(z[:, 0])) + t_fn(z[:, 0])
    return y

grid_coupling = coupling_layer(grid)
plot_grid_transform(axes[0, 2], grid, grid_coupling, n, 'Coupling Layer')

# Show Jacobian determinants
for i, (name, grid_t, det_fn) in enumerate([
    ('Affine det(J)', grid_affine, lambda z: np.full(len(z), np.abs(np.linalg.det(A)))),
    ('Planar det(J)', grid_planar, planar_det),
    ('Coupling det(J)', grid_coupling, lambda z: np.exp(s_fn(z[:, 0]))),
]):
    ax = axes[1, i]
    det_vals = det_fn(grid)
    sc = ax.scatter(grid_t[:, 0], grid_t[:, 1], c=det_vals, cmap='RdYlBu_r', s=5, alpha=0.7)
    plt.colorbar(sc, ax=ax)
    ax.set_title(name)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()

print("Red = stretching (det > 1, density decreases)")
print("Blue = compressing (det < 1, density increases)")

# (d) Stack 4 coupling layers
def make_coupling(seed):
    rng = np.random.RandomState(seed)
    a, b, c = rng.randn(3) * 0.5
    s = lambda x: a * np.sin(b * x + c)
    t = lambda x: rng.randn() * np.cos(rng.randn() * x)
    return s, t

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
z = np.random.randn(3000, 2)
axes[0].scatter(z[:, 0], z[:, 1], s=1, alpha=0.3)
axes[0].set_title('Base: N(0,I)')
axes[0].set_aspect('equal')

for layer in range(4):
    s, t = make_coupling(layer)
    if layer % 2 == 0:
        z[:, 1] = z[:, 1] * np.exp(np.clip(s(z[:, 0]), -2, 2)) + t(z[:, 0])
    else:
        z[:, 0] = z[:, 0] * np.exp(np.clip(s(z[:, 1]), -2, 2)) + t(z[:, 1])

    axes[layer+1].scatter(z[:, 0], z[:, 1], s=1, alpha=0.3)
    axes[layer+1].set_title(f'After layer {layer+1}')
    axes[layer+1].set_aspect('equal')

plt.suptitle('Gaussian → Complex: 4 Coupling Layers', y=1.02)
plt.tight_layout()
plt.show()
```

---

### Solution B3: Train a Normalizing Flow on 2D Data

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# (a) Two moons data
def make_moons(n, noise=0.08):
    t = np.linspace(0, np.pi, n // 2)
    x1 = np.column_stack([np.cos(t), np.sin(t)])
    x2 = np.column_stack([1 - np.cos(t), -np.sin(t) + 0.5])
    data = np.vstack([x1, x2]) + np.random.randn(n, 2) * noise
    return data

data = make_moons(2000)

# (b) Normalizing flow with affine coupling layers
class CouplingLayer:
    def __init__(self, mask, hidden=16):
        self.mask = mask  # binary mask
        # Scale network
        self.W1s = np.random.randn(1, hidden) * 0.1
        self.b1s = np.zeros(hidden)
        self.W2s = np.random.randn(hidden, 1) * 0.01
        self.b2s = np.zeros(1)
        # Translate network
        self.W1t = np.random.randn(1, hidden) * 0.1
        self.b1t = np.zeros(hidden)
        self.W2t = np.random.randn(hidden, 1) * 0.01
        self.b2t = np.zeros(1)

    def _s_t(self, x_masked):
        """Compute scale and translate from masked input."""
        h_s = np.tanh(x_masked @ self.W1s + self.b1s)
        s = h_s @ self.W2s + self.b2s
        s = np.clip(s, -3, 3)  # stability

        h_t = np.tanh(x_masked @ self.W1t + self.b1t)
        t = h_t @ self.W2t + self.b2t
        return s, t

    def forward(self, x):
        """x -> y, returns y and log_det_J"""
        x_masked = x[:, self.mask:self.mask+1]
        s, t = self._s_t(x_masked)

        y = x.copy()
        idx = 1 - self.mask  # the transformed dimension
        y[:, idx:idx+1] = x[:, idx:idx+1] * np.exp(s) + t

        log_det = s.sum(axis=1)
        return y, log_det

    def inverse(self, y):
        """y -> x"""
        y_masked = y[:, self.mask:self.mask+1]
        s, t = self._s_t(y_masked)

        x = y.copy()
        idx = 1 - self.mask
        x[:, idx:idx+1] = (y[:, idx:idx+1] - t) * np.exp(-s)
        return x

    def params_and_grads(self):
        return [
            (self.W1s, 'W1s'), (self.b1s, 'b1s'),
            (self.W2s, 'W2s'), (self.b2s, 'b2s'),
            (self.W1t, 'W1t'), (self.b1t, 'b1t'),
            (self.W2t, 'W2t'), (self.b2t, 'b2t'),
        ]

# Build flow with K=8 layers
K = 8
layers = [CouplingLayer(mask=i % 2, hidden=16) for i in range(K)]

# (c) Training
def compute_log_likelihood(x, layers):
    """Compute log p(x) = log p_0(z) + sum log|det J_k|"""
    z = x.copy()
    total_log_det = np.zeros(len(x))

    # Forward through all layers
    for layer in layers:
        z, log_det = layer.forward(z)
        total_log_det += log_det

    # Base distribution log probability
    log_p0 = -0.5 * np.sum(z**2, axis=1) - np.log(2 * np.pi)

    # Note: we computed forward (data -> latent), so we need
    # log p(x) = log p_0(f(x)) + log|det(df/dx)|
    return log_p0 + total_log_det, z

# Simple gradient descent with numerical gradients
lr = 0.001
batch_size = 256
losses = []

for step in range(2000):
    idx = np.random.randint(0, len(data), batch_size)
    batch = data[idx]

    ll, _ = compute_log_likelihood(batch, layers)
    loss = -np.mean(ll)
    losses.append(loss)

    # Numerical gradient update for each layer
    eps = 1e-4
    for layer in layers:
        for param, name in layer.params_and_grads():
            grad = np.zeros_like(param)
            for index in np.ndindex(param.shape):
                param[index] += eps
                ll_plus, _ = compute_log_likelihood(batch, layers)
                param[index] -= 2 * eps
                ll_minus, _ = compute_log_likelihood(batch, layers)
                param[index] += eps
                grad[index] = -(np.mean(ll_plus) - np.mean(ll_minus)) / (2 * eps)
            param -= lr * grad

    if step % 200 == 0:
        print(f"Step {step}: loss = {loss:.4f}")

# (d) Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Generate samples
z_samples = np.random.randn(1000, 2)
x_gen = z_samples.copy()
for layer in reversed(layers):
    x_gen = layer.inverse(x_gen)

axes[0].scatter(data[:, 0], data[:, 1], s=5, alpha=0.3, c='steelblue', label='Real')
axes[0].scatter(x_gen[:, 0], x_gen[:, 1], s=5, alpha=0.3, c='coral', label='Generated')
axes[0].set_title('Generated vs Real Data')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Density contour
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
log_probs, _ = compute_log_likelihood(grid_points, layers)
log_probs = log_probs.reshape(xx.shape)

axes[1].contourf(xx, yy, np.exp(np.clip(log_probs, -10, 5)), levels=20, cmap='viridis')
axes[1].scatter(data[:, 0], data[:, 1], s=2, alpha=0.3, c='white')
axes[1].set_title('Learned Density')
axes[1].grid(True, alpha=0.3)

# Loss curve
axes[2].plot(losses, alpha=0.7)
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Negative Log-Likelihood')
axes[2].set_title('Training Loss')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### Solution B4: Flow Inversion and Latent Space

```python
# Using the trained flow from B3
import numpy as np
import matplotlib.pyplot as plt

# (a) Map data to latent space
test_data = make_moons(500)
latent, _ = compute_log_likelihood(test_data, layers)  # forward gives us latent

# Actually compute latent points
z_test = test_data.copy()
for layer in layers:
    z_test, _ = layer.forward(z_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(test_data[:, 0], test_data[:, 1], s=5, alpha=0.5, c='steelblue')
axes[0].set_title('Data Space')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(z_test[:, 0], z_test[:, 1], s=5, alpha=0.5, c='coral')
axes[1].set_title('Latent Space (should be Gaussian)')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# QQ plot
from scipy.stats import norm
for dim in range(2):
    sorted_z = np.sort(z_test[:, dim])
    theoretical = norm.ppf(np.linspace(0.01, 0.99, len(sorted_z)))
    axes[2].scatter(theoretical, sorted_z, s=3, alpha=0.3, label=f'dim {dim}')
axes[2].plot([-3, 3], [-3, 3], 'k--', alpha=0.5)
axes[2].set_title('QQ-Plot (latent vs N(0,1))')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# (b) Latent space interpolation
idx_a, idx_b = 0, len(test_data) // 2
x_a, x_b = test_data[idx_a], test_data[idx_b]
z_a, z_b = z_test[idx_a], z_test[idx_b]

t_values = np.linspace(0, 1, 20)
interp_latent = np.array([(1-t) * z_a + t * z_b for t in t_values])

# Map back through inverse flow
interp_data = interp_latent.copy()
for layer in reversed(layers):
    interp_data = layer.inverse(interp_data)

plt.figure(figsize=(10, 5))
plt.scatter(test_data[:, 0], test_data[:, 1], s=3, alpha=0.1, c='gray')
plt.plot(interp_data[:, 0], interp_data[:, 1], 'ro-', markersize=5, linewidth=2)
plt.scatter([x_a[0], x_b[0]], [x_a[1], x_b[1]], c='blue', s=100, zorder=5, marker='*')
plt.title('Latent Space Interpolation Mapped to Data Space')
plt.grid(True, alpha=0.3)
plt.show()

# (c) Anomaly detection
in_dist, _ = compute_log_likelihood(test_data, layers)
noise = np.random.randn(500, 2) * 3
out_dist, _ = compute_log_likelihood(noise, layers)

plt.figure(figsize=(10, 5))
plt.hist(in_dist, bins=50, alpha=0.5, density=True, label='In-distribution', color='steelblue')
plt.hist(out_dist, bins=50, alpha=0.5, density=True, label='Out-of-distribution', color='coral')
plt.xlabel('Log-likelihood')
plt.ylabel('Density')
plt.title('Anomaly Detection: In-dist vs Out-of-dist Log-Likelihoods')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("In-distribution data has higher log-likelihood → flow can detect anomalies!")
```
