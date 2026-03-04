# Tutorial 19: Exercises — Diffusion Models

---

## Part A: Theory

### Exercise A1: Forward Process Derivation 🟢 Easy

**(a)** Starting from $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$, write out the reparameterization: $x_t = \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$.

**(b)** Substitute recursively to show that $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$ and $\epsilon \sim \mathcal{N}(0, I)$.
*Hint: Use the fact that the sum of independent Gaussians is Gaussian with variance equal to the sum of variances.*

**(c)** For $\beta_t = 0.01$ and $T = 1000$, compute $\bar{\alpha}_T$. How much signal remains? How much noise?

**(d)** Plot $\sqrt{\bar{\alpha}_t}$ (signal) and $\sqrt{1-\bar{\alpha}_t}$ (noise) as functions of $t$ for the linear schedule with $\beta_1 = 10^{-4}$, $\beta_T = 0.02$.

---

### Exercise A2: Three Parameterizations 🟡 Medium

Given $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$:

**(a)** Show that predicting $\epsilon$ and predicting $x_0$ are equivalent:
$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}$$

**(b)** Show that the score function satisfies:
$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

**(c)** Therefore, predicting noise $\epsilon$ is equivalent to predicting the score (up to a scaling factor). Write the relationship.

**(d)** Which parameterization is best for training? For understanding? For connecting to other frameworks? Discuss trade-offs.

---

### Exercise A3: DDPM Loss from ELBO 🔴 Hard

**(a)** Write the variational lower bound (ELBO) for diffusion models:
$$L_{VLB} = L_T + \sum_{t=2}^T L_{t-1} + L_0$$
Identify each term.

**(b)** Show that $L_{t-1} = D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$.

**(c)** Since both distributions are Gaussian with the same variance $\tilde{\beta}_t$, show that:
$$L_{t-1} = \frac{1}{2\tilde{\beta}_t} \|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2$$

**(d)** Express $\tilde{\mu}_t$ in terms of $x_t$ and $\epsilon$, then show that predicting $\epsilon$ correctly is equivalent to matching $\mu_\theta$ to $\tilde{\mu}_t$.

**(e)** Explain why the simplified loss $L_{simple} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ (without the time-dependent weighting) works better in practice.

---

### Exercise A4: Score Function for Gaussians 🟡 Medium

**(a)** For $p(x) = \mathcal{N}(x; \mu, \sigma^2)$, compute the score $s(x) = \nabla_x \log p(x)$.

**(b)** For a mixture of Gaussians $p(x) = \sum_k w_k \mathcal{N}(x; \mu_k, \sigma_k^2)$, derive the score. Show that it's a weighted average of individual Gaussian scores.

**(c)** Sketch the score function for $p(x) = 0.5\mathcal{N}(-3, 0.5) + 0.5\mathcal{N}(3, 0.5)$. Where does the score point? Where is it zero?

**(d)** Explain intuitively why following the score with added noise (Langevin dynamics) produces samples from $p(x)$.

---

### Exercise A5: Noise Schedule Analysis 🟡 Medium

**(a)** For the linear schedule $\beta_t = \beta_{min} + \frac{t-1}{T-1}(\beta_{max} - \beta_{min})$, derive $\bar{\alpha}_t$ in closed form (approximately, using $\log(1-x) \approx -x$ for small $x$).

**(b)** For the cosine schedule $\bar{\alpha}_t = \cos^2(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2})$, compute $\beta_t = 1 - \bar{\alpha}_t / \bar{\alpha}_{t-1}$.

**(c)** Compute and compare the signal-to-noise ratio $\text{SNR}(t) = \bar{\alpha}_t / (1-\bar{\alpha}_t)$ for both schedules. Which gives a more uniform rate of information destruction?

**(d)** Why does a more uniform SNR lead to better generation quality? *Hint: Think about which timesteps the model spends most of its capacity on.*

---

### Exercise A6: DDIM Deterministic Sampling 🔴 Hard

**(a)** In DDPM, the reverse step is stochastic: $x_{t-1} = \mu_\theta + \sigma_t z$. In DDIM, the reverse step is deterministic (set $\sigma_t = 0$). Write the DDIM update rule.

**(b)** Show that DDIM can skip timesteps: if we only use timesteps $\{\tau_1, \tau_2, ..., \tau_S\} \subset \{1, ..., T\}$, the update rule still works. This enables sampling in $S \ll T$ steps.

**(c)** DDIM defines a mapping from noise to data that is **deterministic** and **invertible**. Explain why this makes DDIM equivalent to a normalizing flow.

**(d)** Compare DDIM (deterministic, skipping steps) with DDPM (stochastic, all steps). When might you prefer one over the other?

---

### Exercise A7: Connection to Flow Matching 🔴 Hard

**(a)** The probability flow ODE for diffusion is:
$$dx = \left[-\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t)\nabla_x \log p_t(x)\right] dt$$

Show that this defines a velocity field $v(x,t) = -\frac{\beta(t)}{2}[x + \nabla_x \log p_t(x)]$.

**(b)** In flow matching, the velocity field is learned directly. Show that a flow matching model trained with the VP path $x_t = \sqrt{\bar{\alpha}_t} x_1 + \sqrt{1-\bar{\alpha}_t} x_0$ has the same family of solutions as the diffusion probability flow ODE.

**(c)** What are the advantages of the flow matching perspective over the diffusion perspective? *Hint: Think about path choice and sampling efficiency.*

---

## Part B: Coding

### Exercise B1: Implement the Forward Process 🟢 Easy

```python
import numpy as np
import matplotlib.pyplot as plt

# (a) Implement linear and cosine noise schedules
#     Compute beta_t, alpha_t, alpha_bar_t for T=1000
#
# (b) Implement the forward process:
#     Given x0 and timestep t, sample x_t using:
#     x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon
#
# (c) Using 2D data (two moons), visualize x_t at
#     t = 0, 100, 250, 500, 750, 1000
#     Show the distribution at each timestep
#
# (d) Plot sqrt(alpha_bar_t) and sqrt(1-alpha_bar_t) vs t
#     for both schedules. Also plot the SNR.
```

---

### Exercise B2: Train DDPM on 2D Data 🟡 Medium

```python
# (a) Implement a noise prediction network epsilon_theta(x_t, t):
#     - Input: [x1, x2, t_embedding]
#     - Time embedding: sinusoidal encoding of t
#     - Architecture: MLP with 2-3 hidden layers
#     - Output: [eps1, eps2]
#
# (b) Implement DDPM training:
#     For each step:
#       1. Sample x0 from data
#       2. Sample t ~ Uniform{1, ..., T}
#       3. Sample epsilon ~ N(0, I)
#       4. Compute x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * epsilon
#       5. Loss = ||epsilon - epsilon_theta(x_t, t)||^2
#       6. Update weights
#
# (c) Train for 5000 steps and plot the loss curve
#
# (d) How does this compare to GAN training (Tutorial 16)?
#     Is it more stable? Easier to implement?
```

---

### Exercise B3: Implement DDPM Sampling 🟡 Medium

```python
# Using the trained model from Exercise B2:
#
# (a) Implement the full DDPM sampling loop:
#     Starting from x_T ~ N(0, I), for t = T, T-1, ..., 1:
#       1. Predict epsilon_hat = epsilon_theta(x_t, t)
#       2. Compute mean: mu = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_hat)
#       3. Sample: x_{t-1} = mu + sigma_t * z (z ~ N(0,I), except z=0 for t=1)
#
# (b) Generate 1000 samples and compare with real data
#
# (c) Visualize the denoising trajectory:
#     Show x_t at t = T, 0.75T, 0.5T, 0.25T, 0 for several samples
#
# (d) How many timesteps T are needed for good samples?
#     Try T = 10, 50, 100, 500, 1000.
```

---

### Exercise B4: Implement Langevin Dynamics 🟡 Medium

```python
# Score matching and Langevin dynamics
#
# (a) For a known distribution (mixture of 3 Gaussians),
#     compute the exact score function analytically.
#     Visualize as a vector field.
#
# (b) Implement Langevin dynamics:
#     x_{t+1} = x_t + eta * score(x_t) + sqrt(2*eta) * z
#     Starting from random initial points, run for 1000 steps.
#     Do the particles converge to the target distribution?
#
# (c) Try different step sizes eta = [0.001, 0.01, 0.1, 1.0]
#     Which works best? What happens when eta is too large?
#
# (d) Add noise-conditional score: train a network to predict
#     the score at different noise levels (like NCSN).
#     Compare with the DDPM approach.
```

---

### Exercise B5: Compare Noise Schedules 🟡 Medium

```python
# (a) Train DDPM with linear schedule and cosine schedule
#     on the same 2D data
#
# (b) Compare:
#     - Training loss curves
#     - Sample quality (scatter plot)
#     - SNR profiles
#
# (c) Visualize the noised data at the same SNR values
#     for both schedules. Are the noised distributions different?
#
# (d) Which schedule gives better results? Why?
```

---

### Exercise B6: DDIM Sampling 🔴 Hard

```python
# Using the DDPM model trained in Exercise B2:
#
# (a) Implement DDIM sampling (deterministic, sigma_t = 0):
#     x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat +
#               sqrt(1 - alpha_bar_{t-1}) * epsilon_hat
#     where x0_hat = (x_t - sqrt(1-alpha_bar_t)*epsilon_hat) / sqrt(alpha_bar_t)
#
# (b) Use the SAME trained model but sample with:
#     - DDPM (stochastic, T=1000 steps)
#     - DDIM (deterministic, S=100 steps)
#     - DDIM (deterministic, S=20 steps)
#     - DDIM (deterministic, S=5 steps)
#     Compare sample quality.
#
# (c) DDIM is deterministic: verify that the same initial noise
#     always produces the same output. Modify the noise slightly
#     and observe how the output changes (latent space interpolation).
#
# (d) Implement DDIM inversion: map a data point to its noise.
#     Run forward then inverse. Is reconstruction perfect?
```
