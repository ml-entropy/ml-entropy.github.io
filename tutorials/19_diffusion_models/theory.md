# Tutorial 19: Diffusion Models

## 🎯 Why This Tutorial?

**Prerequisites:** Probability distributions (Tutorial 00), KL divergence (Tutorial 03), VAEs (Tutorial 13), normalizing flows (Tutorial 17), flow matching (Tutorial 18)

Diffusion models are the generative framework behind DALL-E, Stable Diffusion, Midjourney, and Sora. They achieve state-of-the-art sample quality by leveraging a beautifully simple idea: **destroy information by gradually adding noise, then learn to reverse the process**.

The mathematical elegance is striking: the forward process (adding noise) requires no learning, and the reverse process (removing noise) can be learned with a simple denoising objective. This tutorial builds deep intuition for how and why this works.

After this tutorial, you'll understand:
- Why gradually adding noise creates a tractable generative model
- The forward and reverse Markov chains
- How the DDPM training loss connects to the variational bound
- Three equivalent parameterizations: predicting $x_0$, predicting $\epsilon$, predicting the score
- Noise schedules and their effect on generation quality
- Score matching and Langevin dynamics
- The SDE perspective that unifies diffusion and score-based models
- How diffusion connects to flow matching

---

## 1. The Core Idea: Destruction and Reconstruction

### 1.1 Intuition

Imagine a photograph. Copy it, adding a tiny bit of static. Copy the copy, adding more static. Repeat 1000 times. The final copy is pure noise — no trace of the original image remains.

Now imagine you could learn to **reverse each tiny step**. Given a slightly noisy image, remove just a bit of the noise. If you chain 1000 such denoising steps together, starting from pure noise, you reconstruct a realistic image.

**Key insight:** Destroying information (adding noise) is trivial. Learning to reverse each **small** step of destruction is tractable. Chaining many small reversals creates from nothing.

### 1.2 Why This Works

1. **Forward process is fixed**: No learning needed — just add Gaussian noise
2. **Each reverse step is small**: Small noise steps have approximately Gaussian reverse distributions
3. **Denoising is a well-studied problem**: Neural networks are excellent denoisers
4. **The loss is simple**: Just predict the noise that was added (MSE loss)

---

## 2. The Forward Process

### 2.1 The Markov Chain

Starting from data $x_0 \sim q(x_0)$, add noise gradually over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

At each step:
- **Scale down** by $\sqrt{1-\beta_t}$ (slightly shrink the signal)
- **Add noise** with variance $\beta_t$ (add a bit of Gaussian noise)

The $\beta_t \in (0, 1)$ are the **noise schedule** — they control how fast information is destroyed.

### 2.2 Jumping to Any Timestep

A crucial property: we can sample $x_t$ directly from $x_0$ without simulating the chain!

Define:
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

Then:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1-\bar{\alpha}_t) I)$$

Using the **reparameterization trick**:
$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**Interpretation:**
- $\sqrt{\bar{\alpha}_t}$ = signal strength (decreases from 1 to ~0)
- $\sqrt{1-\bar{\alpha}_t}$ = noise strength (increases from ~0 to 1)
- Signal-to-noise ratio: $\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$

### 2.3 Proof of the Direct Sampling Formula

By induction. Base case: $t=1$, $q(x_1|x_0) = \mathcal{N}(\sqrt{\alpha_1} x_0, (1-\alpha_1)I) = \mathcal{N}(\sqrt{\bar{\alpha}_1} x_0, (1-\bar{\alpha}_1)I)$. ✓

Inductive step: assume $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_1$ where $\epsilon_1 \sim \mathcal{N}(0,I)$.

Then $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_2$ where $\epsilon_2 \sim \mathcal{N}(0,I)$, $\epsilon_2 \perp \epsilon_1$.

$$x_t = \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_1) + \sqrt{1-\alpha_t} \epsilon_2$$
$$= \sqrt{\alpha_t \bar{\alpha}_{t-1}} x_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})} \epsilon_1 + \sqrt{1-\alpha_t} \epsilon_2$$

Since $\epsilon_1, \epsilon_2$ are independent Gaussians, their weighted sum is Gaussian with variance:
$$\alpha_t(1-\bar{\alpha}_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t = 1 - \bar{\alpha}_t$$

So $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ where $\epsilon \sim \mathcal{N}(0,I)$. $\blacksquare$

---

## 3. Noise Schedules

### 3.1 Linear Schedule

$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

Typical values: $\beta_{\min} = 10^{-4}$, $\beta_{\max} = 0.02$, $T = 1000$.

### 3.2 Cosine Schedule

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

where $s = 0.008$ is a small offset.

### 3.3 Comparison

| Schedule | Early steps | Late steps | Overall |
|----------|------------|------------|---------|
| **Linear** | Adds noise slowly | Adds noise fast | Wastes early steps |
| **Cosine** | Moderate rate | Moderate rate | More uniform SNR change |

The cosine schedule was introduced to fix the problem that the linear schedule destroys information too quickly in the middle timesteps, leaving both early and late steps underutilized.

---

## 4. The Reverse Process

### 4.1 Reversing the Chain

We want to sample from $q(x_{t-1} | x_t)$ — given a noisy image, recover a slightly less noisy one.

This posterior is intractable (requires knowing the data distribution), so we learn it:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

**Why is this Gaussian?** For small $\beta_t$, the reverse transition $q(x_{t-1} | x_t)$ is approximately Gaussian. This was proven by Feller (1949) and is the theoretical foundation of diffusion models.

### 4.2 The Tractable Posterior

When we condition on $x_0$ (which we have during training), the posterior **is** tractable:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)$$

where:
$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)} \beta_t$$

---

## 5. The Training Objective

### 5.1 Variational Lower Bound (ELBO)

The negative log-likelihood can be bounded:

$$-\log p_\theta(x_0) \leq \mathbb{E}_q\left[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = L_{VLB}$$

This decomposes into:

$$L_{VLB} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T \text{ (prior loss)}} + \sum_{t=2}^T \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{(-\log p_\theta(x_0|x_1))}_{L_0}$$

- $L_T$: compares the final noised distribution to the prior (no learnable parameters)
- $L_{t-1}$: KL between the true posterior and the learned reverse step (this is what we train)
- $L_0$: reconstruction term

### 5.2 Simplified Training Objective

Ho et al. (2020) showed that a **simplified loss** works even better:

$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.

**This is just denoising!** Given a noisy input $x_t$, predict the noise $\epsilon$ that was added.

### 5.3 Why the Simple Loss Works

The KL divergence between two Gaussians with the same variance reduces to an MSE between their means:

$$D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) = \frac{1}{2\sigma_t^2} \|\tilde{\mu}_t - \mu_\theta\|^2$$

Since $\tilde{\mu}_t$ depends on $x_0$, and $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$, predicting $\epsilon$ is equivalent to predicting $\mu_\theta$.

---

## 6. Three Parameterizations

The model can equivalently predict different quantities:

### 6.1 Predict Noise ($\epsilon$-prediction, DDPM)

$$\epsilon_\theta(x_t, t) \approx \epsilon$$

Mean: $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$

### 6.2 Predict Clean Data ($x_0$-prediction)

$$\hat{x}_0 = f_\theta(x_t, t) \approx x_0$$

Mean: $\mu_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} f_\theta + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$

### 6.3 Predict Score ($s$-prediction)

$$s_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t)$$

Connection: $\nabla_{x_t} \log q(x_t | x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$

So: $s_\theta = -\frac{\epsilon_\theta}{\sqrt{1-\bar{\alpha}_t}}$

### 6.4 Equivalence

All three parameterizations are related by simple algebra:

$$\epsilon = -\sqrt{1-\bar{\alpha}_t} \cdot s = \frac{x_t - \sqrt{\bar{\alpha}_t} \hat{x}_0}{\sqrt{1-\bar{\alpha}_t}}$$

---

## 7. Score Matching

### 7.1 The Score Function

The **score** of a distribution is the gradient of its log-density:

$$s(x) = \nabla_x \log p(x)$$

It points toward regions of higher probability — like a compass pointing "uphill."

### 7.2 Score Matching Objective

Directly matching the score is intractable (we don't know $\nabla_x \log p(x)$). But **denoising score matching** is tractable:

$$\mathbb{E}_{q(x_t|x_0)}\left[\|s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0)\|^2\right]$$

Since $\nabla_{x_t} \log q(x_t|x_0) = -\epsilon / \sqrt{1-\bar{\alpha}_t}$, this is equivalent to $\epsilon$-prediction!

### 7.3 Langevin Dynamics

Given the score function, we can sample using Langevin dynamics:

$$x_{t+1} = x_t + \eta \nabla_x \log p(x_t) + \sqrt{2\eta} \, z, \quad z \sim \mathcal{N}(0, I)$$

This converges to samples from $p(x)$ as $\eta \to 0$ and number of steps $\to \infty$.

---

## 8. The SDE Perspective

### 8.1 Forward SDE

The discrete forward process becomes a continuous SDE:

$$dx = f(x, t) dt + g(t) dw$$

For VP-SDE: $f(x,t) = -\frac{1}{2}\beta(t)x$, $g(t) = \sqrt{\beta(t)}$

### 8.2 Reverse SDE

Anderson (1982) showed the reverse-time SDE is:

$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{w}$$

The score $\nabla_x \log p_t(x)$ is the only unknown — once learned, we can reverse the process.

### 8.3 Probability Flow ODE

A deterministic ODE with the same marginals:

$$dx = \left[f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)\right] dt$$

This is equivalent to a **continuous normalizing flow** — connecting back to Tutorial 17!

---

## 9. Guidance

### 9.1 Classifier Guidance

Condition on class $y$ by modifying the score:

$$\nabla_x \log p(x_t | y) = \nabla_x \log p(x_t) + \nabla_x \log p(y | x_t)$$

Train a separate classifier $p(y|x_t)$ on noisy data and add its gradient.

### 9.2 Classifier-Free Guidance

Combine conditional and unconditional scores:

$$\tilde{\epsilon}_\theta = (1 + w) \epsilon_\theta(x_t, t, y) - w \cdot \epsilon_\theta(x_t, t, \emptyset)$$

where $w > 0$ is the guidance weight. Higher $w$ = more adherence to the condition, less diversity.

This is what makes text-to-image models like DALL-E and Stable Diffusion work so well.

---

## 10. DDIM: Deterministic Sampling

### 10.1 The DDIM Update

Song et al. (2021) showed that the same training objective supports a **deterministic** sampling process:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}\hat{x}_0}{\sqrt{1-\bar{\alpha}_t}} + \sigma_t \epsilon$$

Setting $\sigma_t = 0$ gives a deterministic mapping — the **probability flow ODE**.

### 10.2 Advantages

- **Fewer steps**: DDIM can generate good samples in 10-50 steps (vs 1000 for DDPM)
- **Deterministic**: Same noise → same image (useful for interpolation)
- **Invertible**: Can map images to their latent noise representation

---

## 11. Connections and Unification

### 11.1 The Big Picture

| Framework | Learns | Sampling | Steps |
|-----------|--------|----------|-------|
| DDPM | Noise predictor $\epsilon_\theta$ | Stochastic reverse chain | 1000 |
| Score-based | Score $\nabla_x \log p_t$ | Langevin dynamics / reverse SDE | 1000 |
| DDIM | Same as DDPM | Deterministic ODE | 10-50 |
| Flow matching | Velocity $v_\theta$ | ODE with OT path | 5-20 |

All four frameworks are mathematically equivalent (same model, different perspective):
- **DDPM** ↔ **Score-based**: $\epsilon_\theta = -\sqrt{1-\bar{\alpha}_t} \cdot s_\theta$
- **DDIM** ↔ **Probability flow ODE**: same deterministic trajectory
- **Flow matching** ↔ **Diffusion**: different path parameterization, same model class

### 11.2 Information-Theoretic View

From our unifying perspective:
- Forward process = adding entropy (destroying information)
- Reverse process = removing entropy (reconstructing information)
- Score function = gradient of information content
- Training = learning the rate of entropy change at each noise level
- Sampling = systematic entropy removal from noise to data

---

## Summary of Key Equations

| Concept | Equation |
|---------|----------|
| Forward process | $q(x_t\|x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$ |
| Reparameterization | $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ |
| DDPM loss | $L = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ |
| Reverse mean | $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta)$ |
| Score-noise relation | $\nabla_x \log q(x_t\|x_0) = -\epsilon/\sqrt{1-\bar{\alpha}_t}$ |
| Langevin dynamics | $x' = x + \eta s(x) + \sqrt{2\eta}z$ |
| Classifier-free guidance | $\tilde{\epsilon} = (1+w)\epsilon_\theta(x,t,y) - w\epsilon_\theta(x,t,\emptyset)$ |
