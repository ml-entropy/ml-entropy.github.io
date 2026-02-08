# Tutorial 11: Learning Rate — The Most Important Hyperparameter

## Overview
The learning rate controls how much we update weights in response to gradients. Too high → divergence. Too low → slow/stuck. We derive optimal learning rates, understand adaptive methods, and explore learning rate schedules.

---

## Part 1: Gradient Descent Basics

### The Update Rule

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where:
- $\theta$: parameters
- $\eta$: learning rate
- $\nabla_\theta \mathcal{L}$: gradient of loss

### What Learning Rate Controls

- **Too small** ($\eta \ll 1$): 
  - Tiny steps → very slow convergence
  - May get stuck in local minima
  
- **Too large** ($\eta \gg 1$):
  - Overshoots minimum
  - May diverge (loss → ∞)
  
- **Just right**:
  - Converges efficiently
  - Escapes shallow local minima

---

## Part 2: Deriving the Optimal Learning Rate

### For Quadratic Loss

Consider simple quadratic: $\mathcal{L}(\theta) = \frac{1}{2}a\theta^2$

Gradient: $\nabla\mathcal{L} = a\theta$

Update: $\theta_{t+1} = \theta_t - \eta \cdot a\theta_t = (1 - \eta a)\theta_t$

**Convergence condition**: $|1 - \eta a| < 1$

Solving: $0 < \eta < \frac{2}{a}$

**Optimal** (fastest convergence): $\eta^* = \frac{1}{a}$ gives $\theta_1 = 0$ in one step!

### For General Convex Functions

Near minimum, loss is approximately quadratic with Hessian $H$:
$$\mathcal{L}(\theta) \approx \mathcal{L}(\theta^*) + \frac{1}{2}(\theta - \theta^*)^T H (\theta - \theta^*)$$

**Convergence requires**: $\eta < \frac{2}{\lambda_{max}}$

where $\lambda_{max}$ is the largest eigenvalue of $H$.

**Optimal for steepest descent**: 
$$\eta^* = \frac{2}{\lambda_{max} + \lambda_{min}}$$

### The Condition Number Problem

**Condition number**: $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$

For ill-conditioned problems ($\kappa \gg 1$):
- Must use small $\eta$ (limited by $\lambda_{max}$)
- Convergence is slow in directions with small eigenvalues
- This is why vanilla SGD struggles with deep networks!

---

## Part 3: Momentum

### The Problem with Vanilla SGD

In ravines (different curvatures in different directions):
- Oscillates across narrow dimension
- Slow progress along ravine

### Momentum Update

$$v_{t+1} = \beta v_t + \nabla_\theta\mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

where $\beta$ is momentum coefficient (typically 0.9).

### Why It Works

**Physical analogy**: Ball rolling down hill with inertia
- Builds up velocity in consistent gradient direction
- Dampens oscillations in inconsistent directions

**Mathematical view**: Exponential moving average of gradients
$$v_t = \sum_{i=0}^{t} \beta^{t-i} g_i$$

Recent gradients weighted more, but past gradients contribute.

### Nesterov Momentum

Look ahead before computing gradient:
$$v_{t+1} = \beta v_t + \nabla_\theta\mathcal{L}(\theta_t - \eta\beta v_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

**Intuition**: "If I keep going this direction, what will the gradient be there?"

Provides better correction, especially near minima.

---

## Part 4: Adaptive Learning Rate Methods

### The Key Idea

Different parameters need different learning rates:
- Frequent features → smaller LR (already well-tuned)
- Rare features → larger LR (need more updates)

### AdaGrad (2011)

Accumulate squared gradients:
$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

**Problem**: $G_t$ only grows → learning rate → 0

### RMSprop (2012)

Exponential moving average instead:
$$G_t = \beta G_{t-1} + (1-\beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

**Benefit**: Learning rate doesn't vanish completely.

### Adam (2014)

Combines momentum + RMSprop:

**First moment** (momentum):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**Second moment** (RMSprop):
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias correction** (crucial for early steps):
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update**:
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Default hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### Deriving Bias Correction

At step 1: $m_1 = (1-\beta_1)g_1$, but we want $\approx g_1$.

Expected value: $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g_t]$

To correct: $\hat{m}_t = \frac{m_t}{1-\beta_1^t}$

---

## Part 5: Learning Rate Schedules

### Why Schedule?

- **Early training**: Large LR to make progress
- **Late training**: Small LR to fine-tune

### Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

Reduce by factor $\gamma$ every $s$ epochs.

### Exponential Decay

$$\eta_t = \eta_0 \cdot e^{-kt}$$

### Cosine Annealing

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{t\pi}{T}\right)$$

Smooth decay from $\eta_{max}$ to $\eta_{min}$ over $T$ steps.

### Warmup

Start with small LR, increase to target:
$$\eta_t = \eta_{target} \cdot \frac{t}{T_{warmup}}$$

**Why warmup?**
- Early gradients are unreliable (random weights)
- Adam's moving averages need time to stabilize
- Large early steps can destabilize training

### One-Cycle Policy

1. **Warmup**: $\eta_{min} \to \eta_{max}$
2. **Annealing**: $\eta_{max} \to \eta_{min}$

Often achieves better results in fewer epochs.

---

## Part 6: The Loss Landscape Perspective

### Sharp vs Flat Minima

- **Sharp minimum**: High curvature, poor generalization
- **Flat minimum**: Low curvature, better generalization

**Learning rate affects which minima we find**:
- Large LR → escapes sharp minima → finds flat minima
- Small LR → gets stuck in nearest minimum (possibly sharp)

### Stochastic Gradient Noise

Mini-batch gradients are noisy estimates:
$$g_{batch} = g_{true} + \text{noise}$$

Noise scale ∝ $\frac{\eta}{\text{batch size}}$

This noise helps **escape local minima** and **find flat minima**.

### The Generalization Connection

Flat minima generalize better because:
- Small parameter perturbations → small loss changes
- Test distribution ≠ training distribution (perturbation)
- Network robust to perturbations → good generalization

**Large learning rate → implicit regularization!**

---

## Part 7: Practical Guidelines

### Choosing Initial Learning Rate

**Learning rate range test**:
1. Train for a few epochs with LR increasing exponentially
2. Plot loss vs LR
3. Choose LR where loss decreases fastest (before divergence)

### Rules of Thumb

| Optimizer | Typical Starting LR |
|-----------|-------------------|
| SGD | 0.1 |
| SGD + Momentum | 0.01 - 0.1 |
| Adam | 0.001 |
| AdamW | 0.001 - 0.0001 |

### Scaling with Batch Size

**Linear scaling rule**: When batch size × $k$, LR × $k$

$$\eta_{new} = \eta_{base} \cdot \frac{\text{batch}_{new}}{\text{batch}_{base}}$$

**Why?** Larger batch → less noise → can take larger steps.

**Caveat**: Needs warmup for very large batches.

---

## Part 8: Information-Theoretic View

### Learning Rate as Information Flow

$$\Delta\theta = -\eta\nabla\mathcal{L}$$

Information content of update:
$$I(\Delta\theta) \propto \log\|\Delta\theta\| = \log\eta + \log\|\nabla\mathcal{L}\|$$

**Learning rate controls bits per update**:
- High LR = more bits changed per step
- Low LR = fewer bits changed per step

### Entropy and Exploration

Large LR + noise → **high entropy** trajectory through parameter space

This connects to:
- Simulated annealing (temperature)
- Exploration vs exploitation

### The MDL Perspective

Minimum Description Length: Best model = shortest description.

Learning rate affects **effective model complexity**:
- Large LR → can't fit fine details → simpler model
- Small LR → fits everything → potentially overfit

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| Vanilla SGD | $\eta < 2/\lambda_{max}$ for convergence |
| Momentum | Averages gradients → faster + less oscillation |
| Adam | Per-parameter adaptive LR |
| Schedules | Large → small over training |
| Warmup | Stabilizes early training |
| Large LR | Implicit regularization (flat minima) |

**The key insight**: Learning rate isn't just about speed — it fundamentally affects which solutions we find and how well they generalize.
