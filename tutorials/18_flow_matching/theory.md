# Tutorial 18: Flow Matching

## 🎯 Why This Tutorial?

**Prerequisites:** Normalizing flows (Tutorial 17), probability distributions (Tutorial 00), basic calculus (derivatives, integrals)

Flow matching is a modern generative modeling framework that elegantly bridges normalizing flows and diffusion models. It answers a deceptively simple question: **what if we could learn a velocity field that moves noise into data?**

The breakthrough insight is that we can train this velocity field with a simple regression loss — no adversarial training, no variational bounds, no ODE simulation during training. Just predict which direction each point should move.

After this tutorial, you'll understand:
- How continuous normalizing flows use ODEs to transform distributions
- Why simulating ODEs during training is expensive and unnecessary
- How conditional flow matching makes training as simple as regression
- Why linear interpolation paths correspond to optimal transport
- The deep connection between flow matching and diffusion models
- Why flow matching has become the foundation for state-of-the-art generative models

---

## 1. From Discrete Flows to Continuous Flows

### 1.1 Recap: Discrete Normalizing Flows

In Tutorial 17, we composed **discrete** transformation steps:

$$z_K = f_K \circ f_{K-1} \circ \cdots \circ f_1(z_0)$$

**What if we take infinitely many infinitesimal steps?** We get a **continuous-time** flow defined by an ordinary differential equation (ODE).

### 1.2 Continuous Normalizing Flows (CNFs)

A CNF defines a velocity field $v_\theta(x, t)$ that describes how points move through space:

$$\frac{dx(t)}{dt} = v_\theta(x(t), t), \quad t \in [0, 1]$$

- At $t=0$: samples start as noise, $x(0) \sim p_0$ (typically $\mathcal{N}(0, I)$)
- At $t=1$: samples arrive at data, $x(1) \sim p_1 \approx p_{data}$

**Think of it like a river:** every point in space has a velocity at each time $t$. If you place a particle (sample) at time 0, it follows the velocity field to time 1. The collection of all particles transforms the noise distribution into the data distribution.

### 1.3 The Continuity Equation

As particles move, probability density evolves according to the **continuity equation**:

$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot (p_t(x) \cdot v_t(x)) = 0$$

This is **conservation of probability** — probability is neither created nor destroyed, just moved around. It's the continuous analog of the change-of-variables formula from discrete flows.

For the log-density, this gives the **instantaneous change of variables**:

$$\frac{d}{dt} \log p_t(x(t)) = -\nabla \cdot v_t(x(t)) = -\text{tr}\left(\frac{\partial v_t}{\partial x}\right)$$

---

## 2. The Problem with CNFs

### 2.1 Training is Expensive

To train a CNF, we need to:
1. **Forward pass**: Solve the ODE $dx/dt = v_\theta(x, t)$ from $t=0$ to $t=1$ (many neural network evaluations)
2. **Compute likelihood**: Integrate $\text{tr}(\partial v / \partial x)$ along the trajectory (expensive trace computation)
3. **Backward pass**: Solve the adjoint ODE backward for gradients (another full ODE solve)

Each training step requires solving two ODEs, each involving dozens to hundreds of neural network evaluations. This is extremely slow.

### 2.2 The Key Question

Can we learn the velocity field $v_\theta$ **without** solving any ODEs during training?

**Yes!** This is exactly what flow matching achieves.

---

## 3. Flow Matching

### 3.1 The Idea

Instead of simulating the ODE, define the target velocity field **directly** and train a neural network to match it.

Given a probability path $p_t(x)$ that interpolates from $p_0$ (noise) to $p_1$ (data), there exists a unique velocity field $u_t(x)$ that generates this path via the continuity equation.

The **flow matching** objective:

$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t \sim U(0,1), x \sim p_t}\left[\|v_\theta(x, t) - u_t(x)\|^2\right]$$

**Problem:** Computing $u_t(x)$ requires knowing the marginal density $p_t(x)$, which is generally intractable.

### 3.2 Conditional Flow Matching — The Breakthrough

**Key insight:** We don't need to match the marginal velocity field. Instead, match **conditional** velocity fields.

For each data point $x_1$, define a simple conditional path from noise $x_0$ to data $x_1$:

$$\psi_t(x_0 | x_1) = (1 - t) \cdot x_0 + t \cdot x_1$$

This is just **linear interpolation** from noise to data!

The conditional velocity field is trivially:

$$u_t(x | x_1) = x_1 - x_0$$

The **conditional flow matching** (CFM) loss:

$$\mathcal{L}_{CFM}(\theta) = \mathbb{E}_{t, x_0 \sim p_0, x_1 \sim p_1}\left[\|v_\theta(\psi_t(x_0|x_1), t) - (x_1 - x_0)\|^2\right]$$

**This is remarkable:** the training algorithm is:
1. Sample $x_1$ from data, $x_0 \sim \mathcal{N}(0, I)$, $t \sim U(0,1)$
2. Compute $x_t = (1-t) x_0 + t x_1$
3. Target velocity: $x_1 - x_0$
4. Loss: $\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$

It's just a **regression problem**! No ODE solving, no adversarial training, no ELBO.

### 3.3 Why Does CFM Work?

**Theorem (Lipman et al., 2023):** The gradient of the CFM loss equals the gradient of the FM loss in expectation:

$$\nabla_\theta \mathcal{L}_{CFM}(\theta) = \nabla_\theta \mathcal{L}_{FM}(\theta)$$

This means optimizing the simple conditional loss is equivalent to optimizing the intractable marginal loss. The proof relies on the tower property of conditional expectation.

---

## 4. Probability Paths

### 4.1 Linear Interpolation (Optimal Transport Path)

$$x_t = (1 - t) x_0 + t x_1$$

Properties:
- **Straight-line paths** from noise to data
- Mean: $\mathbb{E}[x_t] = (1-t) \cdot 0 + t \cdot \mathbb{E}[x_1] = t \cdot \mu_{data}$
- Variance: changes from $I$ to $\text{Var}(x_1)$

### 4.2 Variance-Preserving Path (Diffusion-Style)

$$x_t = \sqrt{1 - t^2} \cdot x_0 + t \cdot x_1$$

Properties:
- The coefficient $\sqrt{1-t^2}$ ensures $\text{Var}(x_t) \approx I$ throughout the path
- Matches the forward process of variance-preserving diffusion models
- Curved paths rather than straight lines

### 4.3 General Gaussian Path

$$x_t = \alpha_t \cdot x_1 + \sigma_t \cdot x_0$$

where $\alpha_0 = 0, \alpha_1 = 1$ and $\sigma_0 = 1, \sigma_1 = 0$. Different choices of $\alpha_t, \sigma_t$ give different paths.

### 4.4 Why Linear Paths Are Best

Linear interpolation corresponds to **optimal transport** — the minimum-cost way to move probability mass. The key advantages:

1. **Straighter trajectories**: Particles take the shortest path from noise to data
2. **Easier to learn**: The velocity field is simpler (straighter = less variation in $v$)
3. **Fewer ODE steps**: At inference, straight paths require fewer integration steps to follow accurately
4. **Better sample quality**: Less numerical error from ODE integration

---

## 5. Sampling (Inference)

At inference time, we **do** solve the ODE:

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

**Euler method** (simplest): divide $[0, 1]$ into $N$ steps of size $\Delta t = 1/N$

**Higher-order methods:** Runge-Kutta (RK4), adaptive step-size solvers (for better accuracy with fewer steps)

The beauty of flow matching: because of the straight OT paths, even Euler with $N=10-20$ steps gives good results. Diffusion models typically need 50-1000 steps!

---

## 6. Connection to Diffusion Models

### 6.1 Score Matching ↔ Flow Matching

Diffusion models learn the **score function** $s_\theta(x, t) = \nabla_x \log p_t(x)$, then use it to define a velocity field for the reverse process.

Flow matching directly learns the **velocity field** $v_\theta(x, t)$.

The relationship:

$$v_t(x) = \frac{\dot{\alpha}_t}{\alpha_t} x + \left(\dot{\sigma}_t - \frac{\dot{\alpha}_t \sigma_t}{\alpha_t}\right) \sigma_t s_t(x)$$

where $s_t(x) = \nabla_x \log p_t(x)$ is the score.

### 6.2 Equivalence of Training Objectives

Under the variance-preserving path, the CFM loss is equivalent (up to a time-dependent weighting) to the denoising score matching loss used in DDPM:

$$\mathcal{L}_{DSM} = \mathbb{E}_{t, x_0, x_1}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right]$$

The difference is purely in **parameterization**: predicting velocity vs. predicting noise vs. predicting score.

### 6.3 Key Differences in Practice

| Aspect | Diffusion Models | Flow Matching |
|--------|-----------------|---------------|
| Training target | Noise $\epsilon$ or score $\nabla \log p$ | Velocity $v$ |
| Path | VP/VE schedule | OT linear path |
| Sampling | SDE or ODE | ODE only |
| Steps needed | 50-1000 | 5-20 |
| Stochasticity | Optional (SDE sampling) | Deterministic (ODE) |
| Training | Simple | Simple |

---

## 7. Rectified Flows

**Rectified flows** (Liu et al., 2023) take the OT idea further: given a trained flow matching model, **straighten** the learned trajectories by retraining.

**Algorithm:**
1. Train a flow matching model
2. Generate $(x_0, x_1)$ pairs by running the ODE
3. Retrain on these pairs with linear interpolation
4. The new paths are straighter → even fewer steps needed

After 2-3 iterations of "reflow," a single-step ($N=1$) ODE integration can produce reasonable samples!

---

## 8. Recent Impact

Flow matching has become the backbone of state-of-the-art generative models:

| Model | Year | Key Contribution |
|-------|------|-----------------|
| **Stable Diffusion 3** | 2024 | Uses rectified flow matching |
| **Flux** | 2024 | Flow matching with improved architecture |
| **Sora** | 2024 | Video generation with flow matching |
| **Stable Audio** | 2024 | Audio generation |

The simplicity and effectiveness of flow matching make it the preferred framework for modern generative AI.

---

## 9. Connection to Information Theory

From our information-theoretic perspective:

- **Velocity field** = the gradient of information transport
- **OT path** = minimum entropy production path
- **Training** = learning the most efficient way to remove entropy from noise
- **Sampling** = following the learned entropy removal path
- The flow moves probability mass along the geodesic in Wasserstein space

---

## Summary of Key Equations

| Concept | Equation |
|---------|----------|
| CNF ODE | $dx/dt = v_\theta(x, t)$ |
| Continuity equation | $\partial p_t / \partial t + \nabla \cdot (p_t v_t) = 0$ |
| Linear interpolation | $x_t = (1-t)x_0 + tx_1$ |
| Conditional velocity | $u_t = x_1 - x_0$ |
| CFM loss | $\mathbb{E}[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2]$ |
| VP path | $x_t = \sqrt{1-t^2} x_0 + t x_1$ |
| Euler sampling | $x_{t+\Delta t} = x_t + v_\theta(x_t, t) \Delta t$ |
