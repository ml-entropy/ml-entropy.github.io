# Tutorial 08: Backpropagation — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Chain Rule Practice

$f(x) = \sin(x^2 + 3x)$

a) **Inner**: $g(x) = x^2 + 3x$, **Outer**: $h(u) = \sin(u)$

b) Chain rule: $\frac{df}{dx} = \frac{dh}{dg} \cdot \frac{dg}{dx}$
$$\frac{df}{dx} = \cos(x^2 + 3x) \cdot (2x + 3)$$

c) At $x = 1$:
- Analytical: $\cos(4) \cdot 5 = -3.27$
- Numerical: $\frac{f(1.001) - f(0.999)}{0.002} \approx -3.27$ ✓

---

### Solution A2 — Single Neuron Gradient

Given: $z = wx + b$, $a = \sigma(z)$, $L = (a - y)^2$

a) **Gradient w.r.t. w**:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
$$= 2(a - y) \cdot \sigma'(z) \cdot x = 2(a-y)\sigma(z)(1-\sigma(z))x$$

b) **Gradient w.r.t. b**:
$$\frac{\partial L}{\partial b} = 2(a-y)\sigma(z)(1-\sigma(z)) \cdot 1$$

c) **Saturation problem**: When $\sigma(z) \approx 0$ or $\sigma(z) \approx 1$:
- $\sigma'(z) = \sigma(1-\sigma) \approx 0$
- Gradients vanish → learning stops!
- This is the **vanishing gradient problem**.

---

### Solution A3 — Softmax Gradient

$\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$

a) **Same index** ($i = k$):
$$\frac{\partial \hat{y}_k}{\partial z_k} = \frac{e^{z_k} \cdot S - e^{z_k} \cdot e^{z_k}}{S^2} = \hat{y}_k - \hat{y}_k^2 = \hat{y}_k(1 - \hat{y}_k)$$

b) **Different index** ($i \neq k$):
$$\frac{\partial \hat{y}_k}{\partial z_i} = \frac{0 - e^{z_k} \cdot e^{z_i}}{S^2} = -\hat{y}_k \hat{y}_i$$

c) **Jacobian matrix**:
$$J_{ki} = \frac{\partial \hat{y}_k}{\partial z_i} = \hat{y}_k(\delta_{ki} - \hat{y}_i) = \text{diag}(\hat{y}) - \hat{y}\hat{y}^T$$

---

### Solution A4 — Cross-Entropy + Softmax

$$L = -\sum_k y_k \log \hat{y}_k$$

$$\frac{\partial L}{\partial z_i} = -\sum_k y_k \frac{1}{\hat{y}_k} \frac{\partial \hat{y}_k}{\partial z_i}$$

Using results from A3:
$$= -\sum_k y_k \frac{1}{\hat{y}_k} \hat{y}_k(\delta_{ki} - \hat{y}_i)$$
$$= -\sum_k y_k (\delta_{ki} - \hat{y}_i)$$
$$= -y_i + \hat{y}_i \sum_k y_k$$

Since $\sum_k y_k = 1$ (one-hot):
$$\boxed{\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i}$$ ∎

---

### Solution A5 — Backprop Recurrence

For layer $l$: $z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}$, $a^{(l)} = \text{ReLU}(z^{(l)})$

Define $\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$.

By chain rule:
$$\delta^{(l)} = \frac{\partial L}{\partial a^{(l)}} \odot \frac{\partial a^{(l)}}{\partial z^{(l)}}$$

From next layer:
$$\frac{\partial L}{\partial a^{(l)}} = \left(W^{(l+1)}\right)^T \delta^{(l+1)}$$

ReLU derivative:
$$\frac{\partial a^{(l)}}{\partial z^{(l)}} = \mathbb{1}[z^{(l)} > 0]$$

Combined:
$$\boxed{\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \mathbb{1}[z^{(l)} > 0]}$$ ∎

---

### Solution A6 — Vanishing Gradient Analysis

a) For sigmoid, $\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$

Gradient at layer 1 involves:
$$\|\delta^{(1)}\| \propto \prod_{l=1}^{L-1} \|W^{(l+1)}\| \cdot \|\sigma'(z^{(l)})\|$$

b) Since $\sigma'(z) \leq 0.25$ always:
$$\|\delta^{(1)}\| \leq C \cdot (0.25)^{49}$$
For $L=50$: $(0.25)^{49} \approx 10^{-30}$ — **gradient is essentially zero!**

c) **ReLU solution**: $\text{ReLU}'(z) = 1$ for $z > 0$
- No multiplication by small numbers
- Gradient passes through unchanged (when active)
- Only problem: "dying ReLU" when $z < 0$ always

---

### Solution A7 — Entropy-Gradient Connection

For sigmoid output $p = \sigma(z)$:
- Derivative: $\sigma'(z) = p(1-p)$
- Binary entropy: $H(p) = -p\log p - (1-p)\log(1-p)$

**To maximize $\sigma'$**: Take derivative w.r.t. $p$:
$$\frac{d}{dp}[p(1-p)] = 1 - 2p = 0 \implies p = 0.5$$

**To maximize $H$**: Take derivative:
$$\frac{dH}{dp} = -\log p + \log(1-p) = 0 \implies p = 0.5$$

Both are maximized at $p = 0.5$ (maximum uncertainty).

**Insight**: High entropy neurons have high gradient flow. The network learns most through uncertain (high-entropy) activations! ∎

---

## Part B: Coding Solutions

### Solution B1 — Numerical vs Analytical Gradient

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Setup
x = 2.0
y = 0.8
w = 0.5

# Forward
z = w * x
a = sigmoid(z)
L = (a - y) ** 2

# Analytical gradient
dL_da = 2 * (a - y)
da_dz = sigmoid_derivative(z)
dz_dw = x
dL_dw_analytical = dL_da * da_dz * dz_dw

# Numerical gradient
eps = 1e-5
def loss_fn(w_test):
    return (sigmoid(w_test * x) - y) ** 2

dL_dw_numerical = (loss_fn(w + eps) - loss_fn(w - eps)) / (2 * eps)

# Compare
print(f"Analytical gradient: {dL_dw_analytical:.8f}")
print(f"Numerical gradient:  {dL_dw_numerical:.8f}")
print(f"Relative error: {abs(dL_dw_analytical - dL_dw_numerical) / (abs(dL_dw_analytical) + 1e-10):.2e}")
```

### Solution B2 — Backward Pass Implementation

```python
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def backward_pass(X, y, W1, b1, W2, b2, z1, a1, z2, a2):
    """
    Compute gradients for 2-layer network.
    
    Args:
        X: input (n_features, m)
        y: one-hot labels (n_classes, m)
        W1, b1: layer 1 parameters
        W2, b2: layer 2 parameters
        z1, a1: layer 1 pre/post activation
        z2, a2: layer 2 pre/post activation (a2 = softmax output)
    
    Returns:
        dW1, db1, dW2, db2: gradients
    """
    m = X.shape[1]
    
    # Output layer: softmax + cross-entropy
    dz2 = a2 - y  # (n_classes, m)
    
    dW2 = (1/m) * dz2 @ a1.T
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    
    # Hidden layer
    da1 = W2.T @ dz2
    dz1 = da1 * relu_derivative(z1)
    
    dW1 = (1/m) * dz1 @ X.T
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2

# Test
np.random.seed(42)
X = np.random.randn(3, 10)  # 3 features, 10 samples
y = np.eye(2)[:, np.random.randint(0, 2, 10)]  # 2 classes

W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros((4, 1))
W2 = np.random.randn(2, 4) * 0.1
b2 = np.zeros((2, 1))

# Forward
z1 = W1 @ X + b1
a1 = relu(z1)
z2 = W2 @ a1 + b2
a2 = softmax(z2)

# Backward
dW1, db1, dW2, db2 = backward_pass(X, y, W1, b1, W2, b2, z1, a1, z2, a2)

print("Gradient shapes:")
print(f"dW1: {dW1.shape}, db1: {db1.shape}")
print(f"dW2: {dW2.shape}, db2: {db2.shape}")
```

### Solution B3 — Gradient Checking

```python
import numpy as np

def gradient_check(model, X, y, eps=1e-7):
    """
    Check analytical gradients against numerical gradients.
    
    Args:
        model: neural network with forward(), backward() methods
        X: input data
        y: labels
        eps: perturbation size
    
    Returns:
        max_error: maximum relative error
    """
    # Get analytical gradients
    model.forward(X)
    model.backward(y)
    
    max_error = 0
    
    for l, (W, dW) in enumerate(zip(model.weights, model.dW)):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                # Numerical gradient
                W[i, j] += eps
                loss_plus = model.cross_entropy_loss(model.forward(X), y)
                
                W[i, j] -= 2 * eps
                loss_minus = model.cross_entropy_loss(model.forward(X), y)
                
                W[i, j] += eps  # Restore
                
                numerical = (loss_plus - loss_minus) / (2 * eps)
                analytical = dW[i, j]
                
                # Relative error
                if abs(analytical) + abs(numerical) > 1e-10:
                    error = abs(analytical - numerical) / (abs(analytical) + abs(numerical))
                    max_error = max(max_error, error)
                    
                    if error > 1e-5:
                        print(f"Layer {l}, ({i},{j}): analytical={analytical:.6f}, "
                              f"numerical={numerical:.6f}, error={error:.2e}")
    
    return max_error

# Example usage (with NeuralNetwork class from notebook)
# max_err = gradient_check(nn, X_test, y_test)
# print(f"Max relative error: {max_err:.2e}")
```

### Solution B4 — Simple Autograd

```python
import numpy as np

class Tensor:
    """A tensor that tracks computation graph for automatic differentiation."""
    
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=True, _children=(self, other), _op='+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=True, _children=(self, other), _op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, n):
        out = Tensor(self.data ** n, requires_grad=True, _children=(self,), _op=f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n-1)) * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=True, _children=(self,), _op='relu')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, requires_grad=True, _children=(self,), _op='sigmoid')
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

# Test
x = Tensor([2.0], requires_grad=True)
w = Tensor([3.0], requires_grad=True)
b = Tensor([1.0], requires_grad=True)

# Forward: y = sigmoid(w*x + b)
z = w * x + b
y = z.sigmoid()

# Backward
y.backward()

print("Forward: y =", y.data)
print("Gradients:")
print(f"  dL/dw = {w.grad}")
print(f"  dL/dx = {x.grad}")
print(f"  dL/db = {b.grad}")
```

### Solution B5 — Gradient Flow Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def analyze_gradient_flow(layer_sizes, activation='relu', n_samples=100):
    """
    Visualize gradient magnitudes through a deep network.
    """
    np.random.seed(42)
    
    # Initialize weights
    weights = []
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2 / layer_sizes[i])
        weights.append(w)
    
    # Random input
    X = np.random.randn(layer_sizes[0], n_samples)
    
    # Forward pass - store activations and pre-activations
    activations = [X]
    z_values = []
    
    a = X
    for w in weights[:-1]:
        z = w @ a
        z_values.append(z)
        if activation == 'relu':
            a = np.maximum(0, z)
        else:  # sigmoid
            a = 1 / (1 + np.exp(-z))
        activations.append(a)
    
    # Output layer (no activation)
    z = weights[-1] @ a
    z_values.append(z)
    activations.append(z)
    
    # Backward pass
    delta = np.ones_like(z)  # Assume gradient from loss is 1
    gradient_magnitudes = [np.mean(np.abs(delta))]
    
    for i in range(len(weights) - 1, 0, -1):
        da = weights[i].T @ delta
        
        if activation == 'relu':
            delta = da * (z_values[i-1] > 0)
        else:  # sigmoid
            s = 1 / (1 + np.exp(-z_values[i-1]))
            delta = da * s * (1 - s)
        
        gradient_magnitudes.insert(0, np.mean(np.abs(delta)))
    
    return gradient_magnitudes

# Compare activations
layer_sizes = [100] + [50] * 15 + [10]  # 15 hidden layers

grad_relu = analyze_gradient_flow(layer_sizes, 'relu')
grad_sigmoid = analyze_gradient_flow(layer_sizes, 'sigmoid')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(grad_relu)+1), grad_relu, 'b-o', label='ReLU')
plt.plot(range(1, len(grad_sigmoid)+1), grad_sigmoid, 'r-s', label='Sigmoid')
plt.xlabel('Layer (from input)')
plt.ylabel('Mean Gradient Magnitude')
plt.title('Gradient Flow Through Deep Network')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(range(1, len(grad_relu)+1), grad_relu, 'b-o', label='ReLU')
plt.semilogy(range(1, len(grad_sigmoid)+1), grad_sigmoid, 'r-s', label='Sigmoid')
plt.xlabel('Layer (from input)')
plt.ylabel('Mean Gradient Magnitude (log scale)')
plt.title('Gradient Flow (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Sigmoid: Gradient at layer 1 / layer 15 = {grad_sigmoid[0]/grad_sigmoid[-1]:.2e}")
print(f"ReLU:    Gradient at layer 1 / layer 15 = {grad_relu[0]/grad_relu[-1]:.2e}")
print("\nSigmoid gradients vanish exponentially; ReLU maintains gradient flow!")
```

---

## Part C: Conceptual Solutions

### C1
**Reverse-mode (backprop) vs forward-mode efficiency:**

- **Forward mode**: Computes $\frac{\partial \text{all outputs}}{\partial \text{one input}}$ in one pass
  - Cost: $O(\text{# inputs})$ passes
  
- **Reverse mode**: Computes $\frac{\partial \text{one output}}{\partial \text{all inputs}}$ in one pass
  - Cost: $O(\text{# outputs})$ passes

Neural networks have **millions of inputs (parameters)** but only **one output (scalar loss)**. Reverse mode wins by a factor of millions!

### C2
**Batch normalization and gradient flow:**

BatchNorm normalizes activations to have zero mean and unit variance:
$$\hat{x} = \frac{x - \mu}{\sigma}$$

**Entropy perspective**:
- Normalization keeps activations in the "linear regime" of sigmoid/tanh
- This is the high-entropy (high-uncertainty) region
- High entropy = high gradient = better learning

**Gradient perspective**:
- Prevents activations from saturating
- Maintains gradient magnitude across layers
- Acts like a "gradient highway"

### C3
**Why optimization is still hard despite exact gradients:**

1. **Non-convexity**: Loss landscape has many local minima, saddle points
2. **Ill-conditioning**: Gradient direction isn't always the best direction
3. **Stochasticity**: Mini-batch gradients are noisy estimates
4. **Sharp minima**: Some minima generalize poorly
5. **Scale sensitivity**: Different parameters need different learning rates

Exact gradients tell us the local slope, but not whether we're heading toward a good minimum!

### C4
**Loss functions with information-theoretic interpretations:**

1. **MSE = Gaussian negative log-likelihood**
   - Assumes noise is Gaussian with fixed variance
   - Minimizing MSE = MLE for Gaussian model

2. **KL Divergence**: Directly measures information loss
   - Used in VAEs, knowledge distillation

3. **Mutual Information**: Used in InfoMax, contrastive learning
   - $I(X; Z) = H(Z) - H(Z|X)$

4. **Focal Loss**: Reweights cross-entropy by prediction confidence
   - Down-weights easy (low-entropy) examples

5. **Label Smoothing**: Adds entropy to hard labels
   - Prevents overconfident predictions

All these connect to the fundamental idea: **learning is compression/communication** — finding efficient representations of data.
