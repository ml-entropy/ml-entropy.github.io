# Tutorial 18: Exercises — Flow Matching

---

## Part A: Theory

### Exercise A1: Continuity Equation 🟢 Easy

The continuity equation is $\frac{\partial p_t}{\partial t} + \nabla \cdot (p_t v_t) = 0$.

**(a)** In 1D, expand the divergence term: $\nabla \cdot (p_t v_t) = \frac{\partial}{\partial x}(p_t v_t)$.
For $p_t(x) = \mathcal{N}(x; \mu_t, \sigma_t^2)$ with $\mu_t = t \cdot \mu_1$ and $\sigma_t = 1 - t + t\sigma_1$, compute $\frac{\partial p_t}{\partial t}$.

**(b)** Given a constant velocity field $v(x) = c$ (all particles move right at speed $c$), verify that $p_t(x) = p_0(x - ct)$ satisfies the continuity equation.

**(c)** For a velocity field $v(x) = -x$ (all particles move toward origin), what happens to a Gaussian $p_0 = \mathcal{N}(0, 1)$? Does density increase or decrease at the origin?

**(d)** Explain why the continuity equation represents "conservation of probability." What would it mean physically if this equation were violated?

---

### Exercise A2: Conditional Velocity Field 🟡 Medium

For the linear interpolation path $x_t = (1-t)x_0 + tx_1$:

**(a)** Compute $\frac{dx_t}{dt}$. Explain why this is the conditional velocity field $u_t(x | x_1) = x_1 - x_0$.

**(b)** For a specific pair $(x_0, x_1) = ((-2, 0), (1, 3))$, compute and plot the trajectory $x_t$ for $t \in [0, 1]$.

**(c)** Show that the trajectory is a straight line in $\mathbb{R}^n$. Why is this related to optimal transport?

**(d)** If we used a quadratic interpolation $x_t = (1-t^2)x_0 + t^2 x_1$, what would the conditional velocity be? Would the paths still be straight?

---

### Exercise A3: Comparing Interpolation Schedules 🟡 Medium

Consider three probability paths from noise $x_0 \sim \mathcal{N}(0, I)$ to data $x_1$:

1. **Linear (OT):** $x_t = (1-t)x_0 + tx_1$
2. **Variance-preserving:** $x_t = \sqrt{1-t^2} x_0 + tx_1$
3. **Cosine:** $x_t = \cos(\pi t / 2) x_0 + \sin(\pi t / 2) x_1$

**(a)** For each path, compute the conditional velocity $u_t = dx_t/dt$.

**(b)** For each path, compute $\mathbb{E}[\|x_t\|^2]$ assuming $x_0 \sim \mathcal{N}(0, I)$ and $\|x_1\|^2 = d$ (data has unit variance per dimension). Which path keeps the variance most constant?

**(c)** Compute the "straightness" of each path, defined as:
$$S = \frac{\|x_1 - x_0\|^2}{\int_0^1 \|dx_t/dt\|^2 dt}$$
$S = 1$ means perfectly straight. Which path is straightest?

**(d)** Why does a straighter path lead to fewer required ODE steps at inference?

---

### Exercise A4: CFM Equals FM 🔴 Hard

**(a)** Write out the flow matching (FM) loss:
$$\mathcal{L}_{FM} = \mathbb{E}_{t, x \sim p_t}[\|v_\theta(x,t) - u_t(x)\|^2]$$
where $u_t(x)$ is the marginal velocity field.

**(b)** Write out the conditional flow matching (CFM) loss:
$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_0, x_1}[\|v_\theta(x_t, t) - u_t(x|x_1)\|^2]$$

**(c)** Expand both losses and show that their gradients with respect to $\theta$ are equal:
$$\nabla_\theta \mathcal{L}_{CFM} = \nabla_\theta \mathcal{L}_{FM}$$
*Hint: Use the tower property $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X|Y]]$ and the fact that the marginal velocity is $u_t(x) = \mathbb{E}[u_t(x|x_1) | x_t = x]$.*

**(d)** Why is this result so important practically? What makes CFM tractable but FM intractable?

---

### Exercise A5: Connection to Diffusion 🔴 Hard

**(a)** In a diffusion model with variance-preserving (VP) schedule, the forward process is:
$$x_t = \sqrt{\bar{\alpha}_t} x_1 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

Rewrite this in the flow matching notation with $x_0 = \epsilon$ and express $\alpha_t, \sigma_t$ in terms of $\bar{\alpha}_t$.

**(b)** The DDPM training loss is $\|\epsilon_\theta(x_t, t) - \epsilon\|^2$. Show that this is equivalent to the CFM loss with the VP path (up to a time-dependent weighting).

**(c)** Given a trained noise predictor $\epsilon_\theta$, derive the corresponding velocity field $v_\theta$.

**(d)** Explain intuitively why flow matching with OT paths needs fewer sampling steps than diffusion with VP paths. *Hint: Think about path curvature.*

---

### Exercise A6: Euler Method Analysis 🟡 Medium

**(a)** Write out the Euler method for solving $dx/dt = v_\theta(x, t)$ with $N$ steps.

**(b)** For a linear velocity field $v(x, t) = a \cdot x + b(t)$, show that the Euler method introduces error proportional to $(\Delta t)^2$ per step.

**(c)** If the true trajectories are perfectly straight (constant velocity), show that Euler with $N=1$ step is exact.

**(d)** This is why OT paths are so valuable: straighter trajectories mean fewer steps needed. For a flow matching model trained on two moons data, estimate how many Euler steps might be needed for good samples with OT paths vs. VP paths.

---

## Part B: Coding

### Exercise B1: Implement Euler ODE Solver 🟢 Easy

```python
import numpy as np
import matplotlib.pyplot as plt

# Implement the Euler method for solving dx/dt = v(x, t)
#
# (a) Write a function euler_solve(v_fn, x0, t_start, t_end, n_steps)
#     that integrates the ODE using Euler's method.
#
# (b) Test on a known ODE: dx/dt = -x (exponential decay)
#     Compare Euler solution with exact solution x(t) = x0 * exp(-t)
#     for N = 5, 10, 50, 200 steps. Plot error vs N.
#
# (c) Test on a 2D circular ODE: dx/dt = -y, dy/dt = x
#     (solution is a circle). Compare Euler with exact solution
#     for different step counts. Does Euler preserve the circle?
#
# (d) Implement RK4 (4th-order Runge-Kutta) and compare with Euler.
#     How many Euler steps are needed to match RK4 with 10 steps?
```

---

### Exercise B2: Train Flow Matching on 2D Data 🟡 Medium

```python
# Implement conditional flow matching from scratch
#
# (a) Generate 2D "two moons" data
#
# (b) Implement a velocity network v_theta(x, t):
#     - Input: [x1, x2, t] (concatenate time)
#     - Architecture: MLP with 2-3 hidden layers
#     - Output: [v1, v2] (2D velocity)
#
# (c) Implement the CFM training loop:
#     For each step:
#       1. Sample x1 from data, x0 ~ N(0,I), t ~ U(0,1)
#       2. Compute x_t = (1-t)*x0 + t*x1
#       3. Target: u = x1 - x0
#       4. Loss: ||v_theta(x_t, t) - u||^2
#       5. Update weights
#
# (d) After training, generate samples:
#     1. Sample x0 ~ N(0, I)
#     2. Solve ODE with Euler: x_{k+1} = x_k + v_theta(x_k, t_k) * dt
#     3. Plot generated samples vs real data
#
# (e) Visualize:
#     - Training loss curve
#     - Learned vector field at t=0, 0.25, 0.5, 0.75, 1.0
#     - Sample trajectories from noise to data
#     - Generated samples vs real data
```

---

### Exercise B3: Compare Sampling Step Counts 🟡 Medium

```python
# Using the trained model from Exercise B2:
#
# (a) Generate samples using Euler with N = 1, 2, 5, 10, 20, 50, 100 steps
#     Plot all results side by side
#
# (b) Compute a quality metric for each:
#     - Simple: compare histograms of generated vs real data
#     - Or: compute Wasserstein distance between generated and real
#
# (c) At what N do the samples become "good enough"?
#     Compare with what you'd expect from a diffusion model (typically N=50+)
#
# (d) Plot sample quality vs number of ODE steps.
#     This demonstrates the practical advantage of straight OT paths.
```

---

### Exercise B4: Variance-Preserving Path 🟡 Medium

```python
# Implement flow matching with the VP path and compare to OT
#
# (a) Implement VP interpolation: x_t = sqrt(1-t^2) * x0 + t * x1
#     Conditional velocity: u_t = t/sqrt(1-t^2) * x0 + x1
#     (derive this by differentiating x_t with respect to t)
#
# (b) Train a flow matching model with the VP path
#     Use the same architecture and data as Exercise B2
#
# (c) Compare OT vs VP:
#     - Plot sample trajectories for both
#     - Are OT trajectories straighter?
#     - Compare sample quality at N=5, 10, 20 Euler steps
#
# (d) Compute path straightness:
#     S = ||x1 - x0||^2 / integral(||dx/dt||^2 dt)
#     Average over many samples. Which path is straighter?
```

---

### Exercise B5: Visualize the Vector Field 🟢 Easy

```python
# Create beautiful visualizations of learned velocity fields
#
# (a) After training flow matching on 2D data, evaluate v_theta(x, t)
#     on a grid for different times t
#
# (b) Create a streamplot at t=0.0, 0.25, 0.5, 0.75, 1.0
#     Color the arrows by their magnitude
#
# (c) Overlay sample trajectories on the vector field
#     Show 20-30 particles flowing from noise to data
#
# (d) Create an animation-style plot: show particles at
#     t=0, 0.1, 0.2, ..., 1.0 as a grid of subplots
#     with the vector field at each time in the background
```
