# Tutorial 09: Regularization - Solutions

## Part A: Theory Solutions

### Solution A1: L2 Gradient
**Given:** $\mathcal{L}_{total} = \mathcal{L}_{data} + \frac{\lambda}{2}\|w\|_2^2$

**Step 1: Compute gradient**
$$\frac{\partial \mathcal{L}_{total}}{\partial w} = \frac{\partial \mathcal{L}_{data}}{\partial w} + \lambda w$$

**Step 2: Weight update rule**
$$w_{t+1} = w_t - \eta\left(\frac{\partial \mathcal{L}_{data}}{\partial w} + \lambda w_t\right)$$

**Step 3: Rearrange**
$$w_{t+1} = w_t(1 - \eta\lambda) - \eta\frac{\partial \mathcal{L}_{data}}{\partial w}$$

The term $(1 - \eta\lambda)$ multiplies the weight directly, "decaying" it toward zero each step. This is why L2 regularization is called **weight decay**.

---

### Solution A2: L1 vs L2 Gradients

**L2 gradient:**
$$\frac{\partial}{\partial w}\left(\frac{\lambda}{2}w^2\right) = \lambda w$$

**L1 gradient:**
$$\frac{\partial}{\partial w}(\lambda|w|) = \lambda \cdot \text{sign}(w) = \begin{cases} \lambda & w > 0 \\ -\lambda & w < 0 \\ \text{undefined} & w = 0 \end{cases}$$

**Why L1 produces sparsity:**
- L2 gradient is proportional to $w$: small weights get small gradients
- L1 gradient is constant ($\pm\lambda$): small weights still get pushed hard toward zero
- L1 has non-smooth point at $w=0$, creating a "barrier" that keeps weights exactly at zero once they reach it

---

### Solution A3: MAP Connection

**Step 1: Posterior**
$$p(w|D) \propto p(D|w) \cdot p(w)$$

**Step 2: Gaussian prior**
$$p(w) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{w^2}{2\sigma^2}\right)$$

**Step 3: Negative log posterior**
$$-\log p(w|D) = -\log p(D|w) - \log p(w) + \text{const}$$
$$= -\log p(D|w) + \frac{w^2}{2\sigma^2} + \text{const}$$

**Step 4: Identification**
Comparing with $\mathcal{L}_{data} + \frac{\lambda}{2}w^2$:
$$\lambda = \frac{1}{\sigma^2}$$

Large $\lambda$ = small $\sigma^2$ = tight prior around zero = strong regularization.

---

### Solution A4: Dropout Expectation

**During training:**
$$\tilde{y} = \sum_i m_i w_i x_i$$
where $m_i \sim \text{Bernoulli}(1-p)$, so $\mathbb{E}[m_i] = 1-p$.

**Expected output:**
$$\mathbb{E}[\tilde{y}] = \sum_i \mathbb{E}[m_i] w_i x_i = (1-p)\sum_i w_i x_i$$

**At test time:**
We use all weights, so to match training expectation:
$$y_{test} = (1-p) \cdot w^T x$$

Or equivalently (inverted dropout): scale during training by $\frac{1}{1-p}$, then use full weights at test time.

---

### Solution A5: Elastic Net

**Gradient:**
$$\nabla \mathcal{L} = \nabla \mathcal{L}_{data} + \lambda_1 \cdot \text{sign}(w) + 2\lambda_2 w$$

**Update rule:**
$$w_{t+1} = w_t - \eta\left(\nabla \mathcal{L}_{data} + \lambda_1 \cdot \text{sign}(w_t) + 2\lambda_2 w_t\right)$$

**Corresponding prior:**
Elastic Net corresponds to a combination of Laplace and Gaussian priors:
$$p(w) \propto \exp(-\lambda_1|w| - \lambda_2 w^2)$$

**Why better:**
- L1 alone can select at most $n$ features when $n < d$
- L2 alone never produces exact zeros
- Elastic Net: sparsity (L1) + stability (L2) + can select groups of correlated features

---

## Part B: Coding Solutions

### Solution B1: L2 Regularization

```python
import numpy as np

def train_with_l2(X, y, lambda_reg, lr=0.01, epochs=100):
    N, D = X.shape
    w = np.zeros(D)
    losses = []
    
    for _ in range(epochs):
        # Predictions
        y_pred = X @ w
        
        # Data loss (MSE)
        data_loss = np.mean((y_pred - y) ** 2)
        
        # Regularization loss
        reg_loss = (lambda_reg / 2) * np.sum(w ** 2)
        
        # Total loss
        total_loss = data_loss + reg_loss
        losses.append(total_loss)
        
        # Gradients
        data_grad = (2 / N) * X.T @ (y_pred - y)
        reg_grad = lambda_reg * w
        
        # Update
        w = w - lr * (data_grad + reg_grad)
        # Or equivalently: w = w * (1 - lr * lambda_reg) - lr * data_grad
    
    return w, losses
```

---

### Solution B2: L1 vs L2 Sparsity

```python
import numpy as np
from sklearn.linear_model import Lasso, Ridge

def compare_regularization_sparsity(X, y, lambdas):
    l1_sparsity = []
    l2_sparsity = []
    
    for lam in lambdas:
        # L1 (Lasso) - sklearn uses alpha = lambda
        lasso = Lasso(alpha=lam, max_iter=10000)
        lasso.fit(X, y)
        l1_nonzero = np.sum(np.abs(lasso.coef_) > 1e-5)
        l1_sparsity.append(l1_nonzero)
        
        # L2 (Ridge)
        ridge = Ridge(alpha=lam)
        ridge.fit(X, y)
        l2_nonzero = np.sum(np.abs(ridge.coef_) > 1e-5)
        l2_sparsity.append(l2_nonzero)
    
    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.semilogx(lambdas, l1_sparsity, 'b-o', label='L1 (Lasso)')
    plt.semilogx(lambdas, l2_sparsity, 'r-o', label='L2 (Ridge)')
    plt.xlabel('λ (regularization strength)')
    plt.ylabel('Number of non-zero weights')
    plt.title('Sparsity: L1 vs L2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return {'l1_sparsity': l1_sparsity, 'l2_sparsity': l2_sparsity}
```

---

### Solution B3: Dropout Layer

```python
import numpy as np

class DropoutLayer:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Create binary mask
        self.mask = (np.random.rand(*x.shape) > self.p).astype(float)
        
        # Scale by 1/(1-p) for inverted dropout
        scale = 1.0 / (1.0 - self.p)
        
        return x * self.mask * scale
    
    def backward(self, grad_output):
        if not self.training or self.p == 0:
            return grad_output
        
        scale = 1.0 / (1.0 - self.p)
        return grad_output * self.mask * scale
```

---

### Solution B4: Regularization Path

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

def plot_regularization_path(X, y, lambda_range):
    n_features = X.shape[1]
    weights = np.zeros((len(lambda_range), n_features))
    
    for i, lam in enumerate(lambda_range):
        model = Ridge(alpha=lam)
        model.fit(X, y)
        weights[i] = model.coef_
    
    plt.figure(figsize=(12, 6))
    for j in range(n_features):
        plt.semilogx(lambda_range, weights[:, j], label=f'w_{j}')
    
    plt.xlabel('λ (regularization strength, log scale)')
    plt.ylabel('Weight value')
    plt.title('Regularization Path: How weights shrink with increasing λ')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage
np.random.seed(42)
X = np.random.randn(100, 5)
true_w = np.array([3, -2, 0, 1, 0])
y = X @ true_w + np.random.randn(100) * 0.5

lambdas = np.logspace(-3, 3, 50)
plot_regularization_path(X, y, lambdas)
```

---

### Solution B5: MC Dropout

```python
import torch
import torch.nn as nn

def enable_dropout(model):
    """Enable dropout layers during evaluation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def mc_dropout_predict(model, x, n_samples=100):
    model.eval()  # Set other layers to eval
    enable_dropout(model)  # But keep dropout active
    
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x)
            predictions.append(pred)
    
    predictions = torch.stack(predictions)  # (n_samples, batch, output_dim)
    
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    
    return mean, std

# Example neural network with dropout
class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```

---

## Part C: Conceptual Answers

### C1: Why L2 Prevents Overfitting
a) **Model complexity**: Large weights allow the model to create sharp, complex decision boundaries. L2 penalizes large weights, preferring smoother functions.

b) **Weight magnitudes**: Overfitted models often have large opposing weights. L2 prevents this by making large weights expensive.

c) **Input sensitivity**: Output change due to input perturbation is $\approx \|w\| \cdot \|\delta x\|$. Smaller weights = less sensitivity to noise.

### C2: Dropout as Ensemble
Each dropout mask creates a different "thinned" network. With $n$ units, there are $2^n$ possible sub-networks. Training with dropout trains all of them (with shared weights). At test time, using scaled weights approximates averaging predictions from all sub-networks. Ensembles reduce variance, improving generalization.

### C3: Different Dropout Rates
- **Hidden layers (p=0.5)**: High capacity, can afford to drop many units
- **Input layer (p=0.1-0.2)**: Dropping inputs loses actual information, not just representations. Too much dropout here = losing signal entirely

### C4: Early Stopping ≈ L2
Gradient descent from zero initialization implicitly regularizes:
- Early: weights are small, model is simple
- Late: weights grow, model becomes complex
- Stopping early ≈ constraining weights to stay small ≈ L2 regularization

Mathematically, for linear regression with gradient descent, the path of solutions is equivalent to the set of L2-regularized solutions with varying $\lambda$.

### C5: Information Bottleneck Perspective
The Information Bottleneck principle seeks representations that:
1. Compress input (minimize $I(X; T)$)
2. Preserve label information (maximize $I(T; Y)$)

Regularization pushes toward simpler representations (compression). It removes information not useful for the task. From this view, regularization implements the "compression" part of the information bottleneck, forcing the network to keep only the most relevant information for prediction.
