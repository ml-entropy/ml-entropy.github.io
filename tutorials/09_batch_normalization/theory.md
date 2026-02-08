# Tutorial 10: Batch Normalization — Stabilizing Deep Networks

## Overview
Batch Normalization (BatchNorm) revolutionized deep learning by enabling training of much deeper networks. We derive the forward and backward passes, understand why it works, and connect it to information flow.

---

## Part 1: The Problem BatchNorm Solves

### Internal Covariate Shift

As training progresses, the distribution of each layer's inputs changes because the previous layers' parameters change. This is called **internal covariate shift**.

**Problems caused**:
1. Each layer must constantly adapt to new input distributions
2. Requires very small learning rates
3. Careful initialization becomes critical
4. Saturating activations (vanishing gradients)

### The Intuition

If we keep each layer's inputs normalized (zero mean, unit variance), training becomes much more stable.

---

## Part 2: The BatchNorm Algorithm

### Forward Pass

For a mini-batch $\{x_1, ..., x_m\}$ at a single neuron:

**Step 1: Compute batch statistics**
$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

**Step 2: Normalize**
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

where $\epsilon$ is small constant for numerical stability.

**Step 3: Scale and shift (learnable parameters)**
$$y_i = \gamma \hat{x}_i + \beta$$

### Why γ and β?

Without them, the network would be limited to zero-mean, unit-variance activations.

$\gamma$ and $\beta$ let the network **undo** normalization if needed:
- If $\gamma = \sigma_B$ and $\beta = \mu_B$, we recover the original $x$
- Network can learn the optimal amount of normalization

### Training vs Inference

**Training**: Use batch statistics $\mu_B$, $\sigma_B^2$

**Inference**: Use **running averages** (computed during training):
$$\mu_{running} \leftarrow (1-\alpha)\mu_{running} + \alpha\mu_B$$
$$\sigma^2_{running} \leftarrow (1-\alpha)\sigma^2_{running} + \alpha\sigma_B^2$$

Typical $\alpha = 0.1$ (momentum).

---

## Part 3: Deriving the Backward Pass

This is surprisingly complex! Let's derive it step by step.

### The Computational Graph

```
x → [μ] → [x - μ] → [÷σ] → x̂ → [×γ + β] → y
         ↓
       [(x-μ)²] → [mean] → [√+ε] → σ
```

### Define Intermediate Variables

Given $\frac{\partial L}{\partial y_i}$ (gradient from upstream), compute gradients w.r.t. $x_i$, $\gamma$, $\beta$.

Let:
- $\hat{x}_i = \frac{x_i - \mu}{\sigma}$ where $\sigma = \sqrt{\sigma^2 + \epsilon}$
- $y_i = \gamma\hat{x}_i + \beta$

### Step 1: Gradients for γ and β

$$\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}$$

### Step 2: Gradient for x̂

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

### Step 3: Gradient for σ²

$$\frac{\partial L}{\partial \sigma^2} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2 + \epsilon)^{-3/2}$$

$$= -\frac{1}{2\sigma^3} \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu)$$

### Step 4: Gradient for μ

$$\frac{\partial L}{\partial \mu} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot \left(-\frac{1}{\sigma}\right) + \frac{\partial L}{\partial \sigma^2} \cdot \frac{-2}{m}\sum_{i=1}^{m}(x_i - \mu)$$

The second term is 0 (sum of deviations from mean):
$$\frac{\partial L}{\partial \mu} = -\frac{1}{\sigma}\sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i}$$

### Step 5: Gradient for x

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\sigma} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{m} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{m}$$

### Simplified Formula

After simplification:
$$\boxed{\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sigma\sqrt{m}} \left( \sqrt{m}\frac{\partial L}{\partial y_i} - \frac{\partial L}{\partial \beta} - \hat{x}_i\frac{\partial L}{\partial \gamma} \right)}$$

---

## Part 4: Why BatchNorm Works

### Theory 1: Reduces Internal Covariate Shift

Original explanation (Ioffe & Szegedy, 2015):
- Normalizing inputs to each layer stabilizes training
- Each layer sees consistent input distribution

### Theory 2: Smooths Loss Landscape

More recent understanding (Santurkar et al., 2018):
- BatchNorm makes the loss surface **smoother**
- Gradients become more predictable (Lipschitz continuous)
- Allows larger learning rates

### Theory 3: Implicit Regularization

- Batch statistics add noise (different batches → different μ, σ)
- Similar effect to dropout
- Prevents overfitting

### Theory 4: Reparameterization

BatchNorm decouples:
- **Direction** of weights (what features to detect)
- **Scale** of activations (handled by γ)

This makes optimization easier.

---

## Part 5: The Entropy/Information Perspective

### Gradient Flow and Entropy

Recall from Tutorial 08: High entropy activations → better gradient flow.

BatchNorm keeps activations **centered** (mean 0) and **scaled** (variance 1):
- Prevents saturation in sigmoid/tanh
- Keeps activations in **high-gradient region**
- Maintains information flow through network

### Information Bottleneck

BatchNorm can be viewed as controlling the **information bottleneck**:
- Normalization removes some information (absolute scale)
- Network must learn **relative** patterns
- This is often more generalizable

### Entropy of Normalized Activations

For Gaussian activations, entropy is:
$$H(X) = \frac{1}{2}\log(2\pi e\sigma^2)$$

BatchNorm sets $\sigma^2 \approx 1$, so:
$$H(\hat{X}) = \frac{1}{2}\log(2\pi e) \approx 1.42 \text{ nats}$$

This **standardizes** the entropy at each layer!

---

## Part 6: Variants of Normalization

### Layer Normalization (LayerNorm)

Normalize across **features** instead of batch:
$$\mu_L = \frac{1}{d}\sum_{j=1}^{d} x_j, \quad \sigma_L^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j - \mu_L)^2$$

**Use case**: RNNs, Transformers (where batch size varies or =1)

### Instance Normalization (InstanceNorm)

Normalize each sample, each channel separately:
$$\mu_{n,c} = \frac{1}{HW}\sum_{h,w} x_{n,c,h,w}$$

**Use case**: Style transfer (removes style information)

### Group Normalization (GroupNorm)

Divide channels into groups, normalize within groups:

**Use case**: Small batch sizes (BatchNorm fails)

### Comparison

| Method | Normalize Over | Batch Independent? | Use Case |
|--------|---------------|-------------------|----------|
| BatchNorm | Batch | No | CNNs with large batches |
| LayerNorm | Features | Yes | Transformers, RNNs |
| InstanceNorm | H, W per channel | Yes | Style transfer |
| GroupNorm | Groups of channels | Yes | Small batch CNNs |

---

## Part 7: Practical Considerations

### Where to Place BatchNorm?

**Option 1**: After linear, before activation
```
x → Linear → BatchNorm → ReLU → ...
```

**Option 2**: After activation
```
x → Linear → ReLU → BatchNorm → ...
```

Both work; Option 1 is more common.

### BatchNorm and Learning Rate

BatchNorm allows **much larger learning rates**:
- Without BN: LR ~ 0.01
- With BN: LR ~ 0.1 or higher

### BatchNorm and Weight Initialization

BatchNorm makes networks **less sensitive** to initialization:
- Bad init → large activations → BN normalizes them
- Training still works (just slower initially)

---

## Summary

**BatchNorm normalizes activations**:
$$y = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

**Why it works**:
1. Reduces internal covariate shift
2. Smooths loss landscape
3. Implicit regularization
4. Maintains gradient flow (entropy perspective)

**The key insight**: By standardizing activations, BatchNorm ensures consistent information flow through deep networks, preventing both vanishing and exploding gradients.
