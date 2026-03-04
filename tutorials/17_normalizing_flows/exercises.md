# Tutorial 17: Exercises — Normalizing Flows

---

## Part A: Theory

### Exercise A1: 1D Change of Variables 🟢 Easy

Let $X \sim \mathcal{N}(0, 1)$ and $Y = f(X)$.

**(a)** For $f(x) = 3x - 2$, find $p_Y(y)$ using the change of variables formula. Verify that $Y \sim \mathcal{N}(-2, 9)$.

**(b)** For $f(x) = x^3$, find $p_Y(y)$. Is the resulting density a named distribution?

**(c)** For $f(x) = e^x$ (so $Y$ is log-normal), derive $p_Y(y)$ and verify it integrates to 1.

**(d)** For $f(x) = \text{sigmoid}(x) = 1/(1+e^{-x})$, derive $p_Y(y)$ for $y \in (0, 1)$. What distribution family does this relate to?

---

### Exercise A2: Jacobian Computation 🟡 Medium

**(a)** For the 2D affine transformation $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ where $A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$, compute the Jacobian matrix and its determinant. How does this transform a unit circle?

**(b)** For $f(x_1, x_2) = (x_1 \cos\theta - x_2 \sin\theta, \; x_1 \sin\theta + x_2 \cos\theta)$ (rotation by angle $\theta$), compute $\det(J)$. Why does a rotation preserve density?

**(c)** For $f(x_1, x_2) = (x_1, \; x_2 + x_1^2)$ (a shear transformation), compute $\det(J)$. Does this change probability densities?

**(d)** For $f(x_1, x_2) = (x_1 \cdot e^{x_2}, \; x_2)$, compute $\det(J)$. Where does the transformation stretch vs. compress space?

---

### Exercise A3: Prove Coupling Layer Properties 🟡 Medium

Consider the affine coupling layer:
$$y_1 = x_1$$
$$y_2 = x_2 \cdot \exp(s(x_1)) + t(x_1)$$

where $s, t: \mathbb{R}^{d/2} \to \mathbb{R}^{d/2}$ are arbitrary functions.

**(a)** Write out the full Jacobian matrix $\frac{\partial(y_1, y_2)}{\partial(x_1, x_2)}$ in block form.

**(b)** Show that this Jacobian is lower-triangular and compute its determinant.

**(c)** Derive the inverse transformation. Show that it requires only one evaluation of $s$ and $t$.

**(d)** If we compose two coupling layers with **alternating masks** (first fixing $x_1$, then fixing $x_2$), write out the composite transformation. Is the composite Jacobian still triangular?

---

### Exercise A4: Log-Likelihood for Composed Flows 🟡 Medium

Consider a flow $f = f_3 \circ f_2 \circ f_1$ where:
- $f_1(z) = Az + b$ (affine, $\det(A) = 2$)
- $f_2(z) = z + u \tanh(w^T z + c)$ (planar flow)
- $f_3$ is a coupling layer with scale function $s$

**(a)** Write the complete log-likelihood formula $\log p(x)$ in terms of the base density $p_0$ and the three Jacobian determinants.

**(b)** For a data point $x = (1, 2)$, describe the steps to compute $\log p(x)$.

**(c)** For training, we compute $\frac{\partial \log p(x)}{\partial \theta}$. Which intermediate quantities must we store for backpropagation?

**(d)** Compare the memory cost of a $K$-layer discrete flow vs. a continuous normalizing flow using the adjoint method.

---

### Exercise A5: Flow Topology 🔴 Hard

**(a)** A normalizing flow maps $\mathbb{R}^n \to \mathbb{R}^n$ via a diffeomorphism. Explain why a single flow transformation cannot map a connected set to a disconnected set (e.g., cannot map a circle to two separate clusters).

**(b)** How do normalizing flows handle multi-modal distributions given this topological constraint? How many coupling layers are needed in principle?

**(c)** Prove that the composition of diffeomorphisms is a diffeomorphism (closure under composition).

**(d)** Consider the density $p(x_1, x_2) = 0.5 \cdot \mathcal{N}((-3, 0), I) + 0.5 \cdot \mathcal{N}((3, 0), I)$. Sketch how a flow might learn to map a standard Gaussian to this bimodal distribution. Where must the flow create high Jacobian determinant, and where low?

---

### Exercise A6: Autoregressive vs Coupling Flows 🔴 Hard

**(a)** Write out the MAF (Masked Autoregressive Flow) transformation for $n=3$:
$$y_i = x_i \cdot \exp(\alpha_i(x_1, ..., x_{i-1})) + \mu_i(x_1, ..., x_{i-1})$$

Show that the Jacobian is lower-triangular and compute its determinant.

**(b)** Write out the IAF (Inverse Autoregressive Flow) transformation. Explain why density evaluation is slow but sampling is fast (opposite of MAF).

**(c)** Show that a coupling layer is a special case of an autoregressive flow where the conditioning functions only depend on the first $d/2$ dimensions.

**(d)** For a model that needs both fast sampling AND fast density evaluation, why might coupling layers be preferred over autoregressive flows?

---

## Part B: Coding

### Exercise B1: 1D Normalizing Flow 🟢 Easy

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Implement a 1D normalizing flow that transforms N(0,1) into a target distribution
#
# (a) Implement the change of variables formula:
#     Given f(x) and f_inv(y), compute p_Y(y)
#
# (b) Apply f(x) = sinh(x) to N(0,1). Plot:
#     - Original N(0,1) density
#     - Transformed density p_Y(y)
#     - Histogram of transformed samples (verify they match p_Y)
#
# (c) Apply f(x) = x + 0.5*sin(2*x). Plot the results.
#     Is this transformation invertible everywhere?
#
# (d) Compose 3 transformations:
#     f1(x) = x + 0.3*sin(x)
#     f2(x) = 1.5*x
#     f3(x) = x + 0.5*tanh(2*x)
#     Plot the density at each intermediate step.
```

---

### Exercise B2: Visualizing 2D Transformations 🟡 Medium

```python
# Visualize how normalizing flow transformations deform 2D space
#
# (a) Create a grid of points in [-3, 3] x [-3, 3]
#     Color each point by its distance from the origin
#     Apply a 2D affine transformation and plot the deformed grid
#     Overlay the Jacobian determinant as a heatmap
#
# (b) Implement a 2D planar flow:
#     f(z) = z + u * tanh(w^T z + b)
#     where u = [0.5, 1], w = [1, 0.5], b = 0
#     Show the deformed grid and the Jacobian determinant field
#
# (c) Implement a 2D coupling layer:
#     y1 = x1
#     y2 = x2 * exp(s(x1)) + t(x1)
#     where s(x1) = 0.5*sin(x1), t(x1) = cos(x1)
#     Show the deformed grid and verify det(J) = exp(s(x1))
#
# (d) Stack 4 coupling layers with alternating masks and
#     random s, t functions. Show how a Gaussian is transformed
#     into a more complex distribution.
```

---

### Exercise B3: Train a Normalizing Flow on 2D Data 🟡 Medium

```python
# Train a normalizing flow to fit 2D data
#
# (a) Generate "two moons" data (or similar 2D dataset)
#
# (b) Implement a normalizing flow with K=8 affine coupling layers:
#     - Alternating masks
#     - s(x) and t(x) as small MLPs (2 -> 16 -> 1)
#     - Base distribution: N(0, I)
#
# (c) Train by maximizing log-likelihood:
#     log p(x) = log p_0(z_0) - sum_k log|det(J_k)|
#     Use Adam-like updates (or simple SGD with momentum)
#
# (d) After training, show:
#     - Learned density as a contour plot
#     - Generated samples vs real data
#     - The base Gaussian mapped through each layer
#     - Log-likelihood curve during training
```

---

### Exercise B4: Flow Inversion and Latent Space 🟡 Medium

```python
# Explore the invertible latent space of a normalizing flow
#
# (a) Using the trained flow from Exercise B3:
#     - Map test data to latent space (inverse flow)
#     - Verify the latent points look Gaussian (QQ-plot or histogram)
#
# (b) Interpolation in latent space:
#     - Take two data points x_a and x_b
#     - Map to latent: z_a = f^{-1}(x_a), z_b = f^{-1}(x_b)
#     - Interpolate: z_t = (1-t)*z_a + t*z_b for t in [0, 1]
#     - Map back: x_t = f(z_t)
#     - Plot the interpolation path in data space
#
# (c) Anomaly detection:
#     - Compute log p(x) for test data (in-distribution)
#     - Compute log p(x) for random noise (out-of-distribution)
#     - Can the flow distinguish them?
```

---

### Exercise B5: Compare Flow Types 🔴 Hard

```python
# Compare different flow architectures on the same data
#
# (a) Implement 3 different flow types:
#     1. Planar flow (K=20 layers)
#     2. Affine coupling flow (K=8 layers)
#     3. Additive coupling flow (no scale, only shift) (K=8 layers)
#
# (b) Train all three on "two moons" data for the same number of steps
#
# (c) Compare:
#     - Final log-likelihood (which fits best?)
#     - Sample quality (scatter plot)
#     - Training speed (wall clock time)
#     - Number of parameters
#
# (d) Discuss: why do coupling layers outperform planar flows
#     despite having a similar theoretical expressiveness?
```
