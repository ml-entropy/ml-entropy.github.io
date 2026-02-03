# Tutorial 10: Batch Normalization - Solutions

## Part A: Theory Solutions

### Solution A1: Forward Pass

**Batch mean:**
$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$

**Batch variance:**
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2$$

**Normalization:**
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

**Final output:**
$$y_i = \gamma \hat{x}_i + \beta$$

---

### Solution A2: Gradient w.r.t. γ

Given $y_i = \gamma \hat{x}_i + \beta$:

$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma}$$

Since $\frac{\partial y_i}{\partial \gamma} = \hat{x}_i$:

$$\boxed{\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i}$$

---

### Solution A3: Gradient w.r.t. $\hat{x}_i$

From $y_i = \gamma \hat{x}_i + \beta$:

$$\frac{\partial y_i}{\partial \hat{x}_i} = \gamma$$

Therefore:
$$\boxed{\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \gamma \cdot \frac{\partial \mathcal{L}}{\partial y_i}}$$

---

### Solution A4: Complete Backward Pass

Let $\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \delta_i$ (the upstream gradient scaled by γ).

**Step 1: Gradient w.r.t. variance**

$$\sigma^2 = \frac{1}{m}\sum_j (x_j - \mu)^2$$
$$\hat{x}_i = (x_i - \mu)(\sigma^2 + \epsilon)^{-1/2}$$

$$\frac{\partial \mathcal{L}}{\partial \sigma^2} = \sum_i \delta_i \cdot (x_i - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2 + \epsilon)^{-3/2}$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \sigma^2} = -\frac{1}{2}\sum_i \delta_i (x_i - \mu)(\sigma^2 + \epsilon)^{-3/2}}$$

**Step 2: Gradient w.r.t. mean**

Two paths: directly through $\hat{x}_i$ and through $\sigma^2$:

Direct path:
$$\frac{\partial \hat{x}_i}{\partial \mu} = -(\sigma^2 + \epsilon)^{-1/2}$$

Through variance:
$$\frac{\partial \sigma^2}{\partial \mu} = \frac{2}{m}\sum_j (x_j - \mu) \cdot (-1) = 0$$ (by definition of mean)

So:
$$\boxed{\frac{\partial \mathcal{L}}{\partial \mu} = -\sum_i \delta_i (\sigma^2 + \epsilon)^{-1/2}}$$

**Step 3: Gradient w.r.t. input**

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial \mathcal{L}}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial x_i} + \frac{\partial \mathcal{L}}{\partial \mu} \cdot \frac{\partial \mu}{\partial x_i}$$

Where:
- $\frac{\partial \hat{x}_i}{\partial x_i} = (\sigma^2 + \epsilon)^{-1/2}$
- $\frac{\partial \sigma^2}{\partial x_i} = \frac{2(x_i - \mu)}{m}$
- $\frac{\partial \mu}{\partial x_i} = \frac{1}{m}$

Combining:
$$\boxed{\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{m\sqrt{\sigma^2 + \epsilon}}\left(m\delta_i - \sum_j \delta_j - \hat{x}_i \sum_j \delta_j \hat{x}_j\right)}$$

---

### Solution A5: Running Statistics

**Why not batch statistics at inference:**
- At inference, we might have batch size = 1 (single sample)
- Batch statistics would be undefined or meaningless
- We want deterministic predictions (same input = same output)

**Exponential moving average:**
$$\mu_{running} = (1-\alpha)\mu_{running} + \alpha \mu_{batch}$$
$$\sigma^2_{running} = (1-\alpha)\sigma^2_{running} + \alpha \sigma^2_{batch}$$

Typical $\alpha$ (momentum) = 0.1

**Effect of momentum:**
- Too high (α → 1): Running stats dominated by recent batches, unstable
- Too low (α → 0): Running stats update too slowly, don't adapt
- α = 0.1: Good balance, exponential average over ~10 batches

---

## Part B: Coding Solutions

### Solution B1: Forward Pass

```python
import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    N, D = x.shape
    
    # Step 1: Batch mean
    mu = np.mean(x, axis=0)  # (D,)
    
    # Step 2: Batch variance
    var = np.var(x, axis=0)  # (D,)
    
    # Step 3: Normalize
    std = np.sqrt(var + eps)
    x_norm = (x - mu) / std  # (N, D)
    
    # Step 4: Scale and shift
    out = gamma * x_norm + beta  # (N, D)
    
    # Cache for backward
    cache = (x, x_norm, mu, var, std, gamma, eps)
    
    return out, cache
```

---

### Solution B2: Backward Pass

```python
def batchnorm_backward(dout, cache):
    x, x_norm, mu, var, std, gamma, eps = cache
    N, D = x.shape
    
    # Gradients w.r.t. gamma and beta
    dgamma = np.sum(dout * x_norm, axis=0)  # (D,)
    dbeta = np.sum(dout, axis=0)  # (D,)
    
    # Gradient w.r.t. x_norm
    dx_norm = dout * gamma  # (N, D)
    
    # Gradient w.r.t. variance
    dvar = np.sum(dx_norm * (x - mu) * (-0.5) * (var + eps)**(-1.5), axis=0)
    
    # Gradient w.r.t. mean
    dmu = np.sum(dx_norm * (-1/std), axis=0) + dvar * np.sum(-2 * (x - mu), axis=0) / N
    
    # Gradient w.r.t. x
    dx = dx_norm / std + dvar * 2 * (x - mu) / N + dmu / N
    
    return dx, dgamma, dbeta
```

---

### Solution B3: Gradient Check

```python
def check_batchnorm_gradient():
    np.random.seed(42)
    
    N, D = 5, 4
    x = np.random.randn(N, D)
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)
    
    # Analytical gradients
    out, cache = batchnorm_forward(x, gamma, beta)
    dx_ana, dgamma_ana, dbeta_ana = batchnorm_backward(dout, cache)
    
    # Numerical gradients
    eps = 1e-5
    
    # Check dx
    dx_num = np.zeros_like(x)
    for i in range(N):
        for j in range(D):
            x[i, j] += eps
            out_plus, _ = batchnorm_forward(x, gamma, beta)
            x[i, j] -= 2 * eps
            out_minus, _ = batchnorm_forward(x, gamma, beta)
            x[i, j] += eps
            dx_num[i, j] = np.sum(dout * (out_plus - out_minus)) / (2 * eps)
    
    # Check dgamma
    dgamma_num = np.zeros_like(gamma)
    for j in range(D):
        gamma[j] += eps
        out_plus, _ = batchnorm_forward(x, gamma, beta)
        gamma[j] -= 2 * eps
        out_minus, _ = batchnorm_forward(x, gamma, beta)
        gamma[j] += eps
        dgamma_num[j] = np.sum(dout * (out_plus - out_minus)) / (2 * eps)
    
    print(f"dx max diff: {np.max(np.abs(dx_ana - dx_num)):.2e}")
    print(f"dgamma max diff: {np.max(np.abs(dgamma_ana - dgamma_num)):.2e}")
    print(f"dbeta max diff: {np.max(np.abs(dbeta_ana - np.sum(dout, axis=0))):.2e}")

check_batchnorm_gradient()
```

---

### Solution B4: Layer Normalization

```python
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.cache = None
    
    def forward(self, x):
        # Normalize across last dimension(s)
        mu = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        
        x_norm = (x - mu) / std
        out = self.gamma * x_norm + self.beta
        
        self.cache = (x, x_norm, mu, var, std)
        return out
    
    def backward(self, dout):
        x, x_norm, mu, var, std = self.cache
        D = x.shape[-1]
        
        dgamma = np.sum(dout * x_norm, axis=tuple(range(len(dout.shape)-1)))
        dbeta = np.sum(dout, axis=tuple(range(len(dout.shape)-1)))
        
        dx_norm = dout * self.gamma
        
        # Similar to BatchNorm but normalize across features
        dvar = np.sum(dx_norm * (x - mu) * (-0.5) * (var + self.eps)**(-1.5), axis=-1, keepdims=True)
        dmu = np.sum(dx_norm * (-1/std), axis=-1, keepdims=True)
        
        dx = dx_norm / std + dvar * 2 * (x - mu) / D + dmu / D
        
        return dx, dgamma, dbeta
```

---

### Solution B5: Visualize Internal Covariate Shift

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def visualize_covariate_shift():
    # Create networks
    class DeepNet(nn.Module):
        def __init__(self, use_bn=False):
            super().__init__()
            layers = []
            dims = [784, 256, 256, 256, 256, 10]
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2:
                    if use_bn:
                        layers.append(nn.BatchNorm1d(dims[i+1]))
                    layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)
            self.hooks = []
            self.activations = []
        
        def forward(self, x):
            return self.net(x.view(x.size(0), -1))
        
        def record_activations(self):
            self.activations = []
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.ReLU):
                    def hook(module, inp, out, layer_idx=i):
                        self.activations.append(out.detach().cpu().numpy().flatten())
                    self.hooks.append(layer.register_forward_hook(hook))
    
    # Generate dummy data
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for use_bn, row in [(False, 0), (True, 1)]:
        model = DeepNet(use_bn=use_bn)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(4):
            model.record_activations()
            out = model(X[:100])
            
            # Plot activations
            if len(model.activations) > 0:
                axes[row, epoch].hist(model.activations[0], bins=50, alpha=0.7)
                axes[row, epoch].set_title(f'{"With BN" if use_bn else "No BN"}, Epoch {epoch}')
                axes[row, epoch].set_xlim(-5, 5)
            
            # Clear hooks
            for h in model.hooks:
                h.remove()
            model.hooks = []
            
            # Train
            for _ in range(50):
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
    
    plt.suptitle('Activation Distributions: Layer 1', fontsize=14)
    plt.tight_layout()
    plt.show()

visualize_covariate_shift()
```

---

## Part C: Conceptual Answers

### C1: Why γ and β?
Without γ and β, the network couldn't represent the identity function. After normalization, outputs are constrained to have mean 0 and variance 1. With γ and β, the network can learn to undo the normalization if needed: γ = σ, β = μ recovers the original distribution. This ensures BatchNorm doesn't reduce model capacity.

### C2: Why ε?
If ε = 0 and variance is exactly 0 (all inputs identical), we'd divide by zero. Even if variance is very small but non-zero, we'd get numerical instability. ε (typically 1e-5) ensures the denominator is always positive and provides numerical stability.

### C3: Higher Learning Rates
BatchNorm smooths the loss landscape by:
1. Reducing dependence between layers (each layer sees stable input distributions)
2. Making gradients more predictable (normalized activations have bounded gradients)
3. Reducing the effect of bad initialization

This smoother landscape allows larger steps without overshooting minima.

### C4: Training vs Inference
a) **Batch statistics can't be used at inference:**
   - Single sample has no meaningful batch statistics
   - Want deterministic predictions (same input = same output)

b) **Running statistics solution:**
   - Maintain exponential moving average during training
   - Use these fixed statistics at inference

c) **Train/test discrepancy:**
   - Running statistics may not match true data statistics
   - Problem with small batches or non-i.i.d. data
   - Can cause accuracy drop when switching to eval mode

### C5: Information-Theoretic View
Sigmoid and tanh derivatives are largest when input is near zero (output near 0.5 for sigmoid). This is the maximum entropy region. BatchNorm centers activations around zero, keeping them in the high-gradient region. This maximizes gradient flow (information flow) during backpropagation.

### C6: Regularization Effect
BatchNorm introduces noise through mini-batch sampling:
- Different batches → different μ and σ² → different normalization
- This noise is similar to dropout: it prevents co-adaptation of neurons
- Acts as implicit regularization, reducing the need for explicit dropout

This is why networks with BatchNorm often need less dropout.
