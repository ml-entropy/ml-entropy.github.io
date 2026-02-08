# Tutorial 09: Regularization - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Derive the gradient of L2-regularized loss**

Given loss $\mathcal{L}_{total} = \mathcal{L}_{data} + \frac{\lambda}{2}\|w\|_2^2$:

1. Compute $\frac{\partial \mathcal{L}_{total}}{\partial w}$
2. Write the weight update rule
3. Show this is equivalent to "weight decay" with factor $(1 - \eta\lambda)$

---

### Exercise A2 🟢 (Easy)
**Compare L1 and L2 gradients**

For a single weight $w$:
1. Compute $\frac{\partial}{\partial w}\left(\frac{\lambda}{2}w^2\right)$ (L2)
2. Compute $\frac{\partial}{\partial w}(\lambda|w|)$ (L1)
3. Explain why L1 produces sparse solutions

---

### Exercise A3 🟡 (Medium)
**Derive the MAP connection**

Show that L2 regularization corresponds to a Gaussian prior:
1. Start with posterior: $p(w|D) \propto p(D|w) \cdot p(w)$
2. Assume Gaussian prior: $p(w) = \mathcal{N}(0, \sigma^2)$
3. Take negative log and show it equals L2-regularized loss
4. Express $\lambda$ in terms of $\sigma^2$

---

### Exercise A4 🟡 (Medium)
**Dropout expectation**

Prove that at test time, scaling weights by $(1-p)$ gives the expected output:
1. During training: $\tilde{y} = \sum_i m_i w_i x_i$ where $m_i \sim \text{Bernoulli}(1-p)$
2. Compute $\mathbb{E}[\tilde{y}]$
3. Show test-time formula $y = (1-p)\sum_i w_i x_i$ matches

---

### Exercise A5 🔴 (Hard)
**Elastic Net derivation**

Elastic Net combines L1 and L2: $\mathcal{L} = \mathcal{L}_{data} + \lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$

1. Derive the gradient
2. Show the update rule
3. What prior does this correspond to?
4. Why might this be better than pure L1 or L2?

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Implement L2 regularization manually**

```python
def train_with_l2(X, y, lambda_reg, lr=0.01, epochs=100):
    """
    Train linear regression with L2 regularization (without using libraries).
    
    Args:
        X: Input features (N, D)
        y: Targets (N,)
        lambda_reg: Regularization strength
        lr: Learning rate
        epochs: Number of training epochs
    
    Returns:
        w: Learned weights
        losses: List of losses per epoch
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B2 🟡 (Medium)
**Compare L1 vs L2 sparsity**

```python
def compare_regularization_sparsity(X, y, lambdas):
    """
    Train models with different regularization and compare weight sparsity.
    
    1. For each lambda in lambdas:
       - Train with L1 (use sklearn Lasso)
       - Train with L2 (use sklearn Ridge)
       - Count non-zero weights (|w| > 1e-5)
    
    2. Plot: lambda vs number of non-zero weights for both L1 and L2
    
    Returns:
        Dictionary with 'l1_sparsity' and 'l2_sparsity' lists
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B3 🟡 (Medium)
**Implement Dropout layer**

```python
class DropoutLayer:
    def __init__(self, p=0.5):
        """
        Args:
            p: Probability of dropping a unit
        """
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """
        Apply dropout during forward pass.
        Remember: scale by 1/(1-p) during training for inverted dropout.
        """
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output):
        """
        Backward pass through dropout.
        """
        # YOUR CODE HERE
        pass
```

---

### Exercise B4 🔴 (Hard)
**Visualize regularization paths**

```python
def plot_regularization_path(X, y, lambda_range):
    """
    Plot how weights change as regularization strength increases.
    
    For each lambda in lambda_range:
    1. Train Ridge regression
    2. Store all weights
    
    Create a plot with:
    - X-axis: log(lambda)
    - Y-axis: weight values
    - One line per feature
    
    This visualizes how weights shrink toward zero.
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B5 🔴 (Hard)
**Implement MC Dropout for uncertainty**

```python
def mc_dropout_predict(model, x, n_samples=100):
    """
    Use Monte Carlo Dropout for uncertainty estimation.
    
    1. Keep dropout active at test time
    2. Run n_samples forward passes
    3. Return mean prediction and uncertainty (std)
    
    Args:
        model: Neural network with dropout layers
        x: Input sample
        n_samples: Number of forward passes
    
    Returns:
        mean: Mean prediction
        std: Standard deviation (uncertainty)
    """
    # YOUR CODE HERE
    pass
```

---

## Part C: Conceptual Questions

### C1 🟢
Why does L2 regularization prevent overfitting? Explain in terms of:
a) Model complexity
b) Weight magnitudes
c) Sensitivity to input perturbations

### C2 🟡
Dropout is often described as "training an ensemble." Explain this interpretation. Why does averaging over many sub-networks help generalization?

### C3 🟡
Why is the dropout rate typically 0.5 for hidden layers but lower (0.1-0.2) for input layers?

### C4 🔴
How does early stopping relate to L2 regularization? (Hint: Consider gradient descent dynamics and the implicit bias toward the origin)

### C5 🔴
From an information-theoretic perspective, how does regularization relate to the Information Bottleneck principle?
