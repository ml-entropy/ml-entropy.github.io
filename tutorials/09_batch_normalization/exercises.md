# Tutorial 10: Batch Normalization - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Forward pass normalization**

Given a mini-batch $\{x_1, ..., x_m\}$:
1. Compute the batch mean: $\mu_B = ?$
2. Compute the batch variance: $\sigma_B^2 = ?$
3. Write the normalization formula: $\hat{x}_i = ?$
4. Write the final output with learnable parameters: $y_i = ?$

---

### Exercise A2 🟡 (Medium)
**Derive $\frac{\partial \mathcal{L}}{\partial \gamma}$**

Given the forward pass $y_i = \gamma \hat{x}_i + \beta$:
1. Express the gradient in terms of upstream gradient $\frac{\partial \mathcal{L}}{\partial y_i}$
2. Show that $\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_i \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i$

---

### Exercise A3 🟡 (Medium)
**Derive $\frac{\partial \mathcal{L}}{\partial \hat{x}_i}$**

Starting from $y_i = \gamma \hat{x}_i + \beta$:
1. Show $\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \gamma \cdot \frac{\partial \mathcal{L}}{\partial y_i}$

---

### Exercise A4 🔴 (Hard)
**Complete backward pass derivation**

The hardest part of BatchNorm backward is computing $\frac{\partial \mathcal{L}}{\partial x_i}$.

Given:
- $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
- $\mu = \frac{1}{m}\sum_j x_j$
- $\sigma^2 = \frac{1}{m}\sum_j (x_j - \mu)^2$

Derive $\frac{\partial \mathcal{L}}{\partial x_i}$ by:
1. Computing $\frac{\partial \mathcal{L}}{\partial \sigma^2}$
2. Computing $\frac{\partial \mathcal{L}}{\partial \mu}$
3. Combining using chain rule

---

### Exercise A5 🔴 (Hard)
**Running statistics derivation**

During inference, we use running mean and variance. Derive:
1. Why we can't use batch statistics at inference
2. The exponential moving average formulas
3. What happens if momentum is too high or too low

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Implement forward pass**

```python
def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """
    Batch normalization forward pass.
    
    Args:
        x: Input, shape (N, D)
        gamma: Scale parameter, shape (D,)
        beta: Shift parameter, shape (D,)
        eps: Small constant for numerical stability
    
    Returns:
        out: Normalized output, shape (N, D)
        cache: Values needed for backward pass
    """
    # YOUR CODE HERE
    # 1. Compute batch mean
    # 2. Compute batch variance
    # 3. Normalize
    # 4. Scale and shift
    pass
```

---

### Exercise B2 🟡 (Medium)
**Implement backward pass**

```python
def batchnorm_backward(dout, cache):
    """
    Batch normalization backward pass.
    
    Args:
        dout: Upstream gradient, shape (N, D)
        cache: Values from forward pass
    
    Returns:
        dx: Gradient w.r.t. input, shape (N, D)
        dgamma: Gradient w.r.t. gamma, shape (D,)
        dbeta: Gradient w.r.t. beta, shape (D,)
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B3 🟡 (Medium)
**Verify with numerical gradient**

```python
def check_batchnorm_gradient():
    """
    Verify backward pass using numerical gradients.
    
    1. Create random input, gamma, beta
    2. Compute analytical gradients using batchnorm_backward
    3. Compute numerical gradients using finite differences
    4. Compare and report max difference
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B4 🔴 (Hard)
**Implement Layer Normalization**

```python
class LayerNorm:
    """
    Layer Normalization (normalize across features, not batch).
    
    Unlike BatchNorm:
    - Normalizes across feature dimension
    - No running statistics needed
    - Works the same at train and test time
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        # YOUR CODE HERE
        pass
    
    def backward(self, dout):
        # YOUR CODE HERE
        pass
```

---

### Exercise B5 🔴 (Hard)
**Visualize internal covariate shift**

```python
def visualize_covariate_shift():
    """
    Train two networks (with and without BatchNorm) and visualize
    how activation distributions change during training.
    
    1. Create a deep network (5+ layers)
    2. Record activation distributions at each layer every N epochs
    3. Plot histograms showing how distributions evolve
    4. Compare with/without BatchNorm
    """
    # YOUR CODE HERE
    pass
```

---

## Part C: Conceptual Questions

### C1 🟢
Why do we need learnable parameters γ and β? What would happen without them?

### C2 🟢
Why is ε needed in the denominator? What would happen if ε = 0?

### C3 🟡
Explain why BatchNorm allows higher learning rates. What's the connection to the loss landscape?

### C4 🟡
BatchNorm behaves differently at training vs inference. Explain:
a) Why batch statistics can't be used at inference
b) How running statistics solve this
c) Why this can cause problems (train/test discrepancy)

### C5 🔴
From an information-theoretic perspective, how does BatchNorm affect the gradient flow? (Hint: Think about where sigmoid/tanh derivatives are largest)

### C6 🔴
Why does BatchNorm have a regularization effect? Explain the connection to the noise introduced by mini-batch sampling.
