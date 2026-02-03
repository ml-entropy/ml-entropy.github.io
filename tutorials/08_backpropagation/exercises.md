# Tutorial 08: Backpropagation — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Chain Rule Practice
For $f(x) = \sin(x^2 + 3x)$:
a) Identify the inner and outer functions
b) Compute $\frac{df}{dx}$ using the chain rule
c) Verify numerically at $x = 1$

### Exercise A2 🟢 — Single Neuron Gradient
For a neuron with $z = wx + b$, $a = \sigma(z)$, $L = (a - y)^2$:
a) Derive $\frac{\partial L}{\partial w}$
b) Derive $\frac{\partial L}{\partial b}$
c) What happens to the gradient when $\sigma(z) \approx 0$ or $\sigma(z) \approx 1$?

### Exercise A3 🟡 — Softmax Gradient Derivation
For softmax $\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$:
a) Compute $\frac{\partial \hat{y}_k}{\partial z_k}$ (same index)
b) Compute $\frac{\partial \hat{y}_k}{\partial z_i}$ for $i \neq k$ (different index)
c) Express in matrix form using the Jacobian

### Exercise A4 🟡 — Cross-Entropy + Softmax Simplification
Prove that for cross-entropy loss $L = -\sum_k y_k \log \hat{y}_k$ with softmax output:
$$\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$$

This remarkably simple form is why this combination is so popular.

### Exercise A5 🔴 — Derive Backprop Recurrence
For a fully-connected network with ReLU activations, derive the backward recurrence:
$$\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \mathbb{1}[z^{(l)} > 0]$$

### Exercise A6 🔴 — Vanishing Gradient Analysis
For a network with $L$ layers and sigmoid activations:
a) Show that the gradient magnitude scales as $\prod_{l=1}^{L} \|\sigma'(z^{(l)})\|$
b) Since $\sigma'(z) \leq 0.25$, what happens for $L = 50$ layers?
c) How does ReLU solve this problem?

### Exercise A7 🔴 — The Entropy-Gradient Connection
Prove: For sigmoid activation, the derivative $\sigma'(z) = \sigma(1-\sigma)$ is maximized when the output entropy $H = -p\log p - (1-p)\log(1-p)$ is maximized.

---

## Part B: Coding

### Exercise B1 🟢 — Numerical vs Analytical Gradient
```python
# TODO: Implement gradient verification
# 1. Define a simple function f(w) = (sigmoid(w*x) - y)^2
# 2. Compute analytical gradient using chain rule
# 3. Compute numerical gradient using finite differences
# 4. Compare and report relative error
```

### Exercise B2 🟡 — Implement Backward Pass
```python
# TODO: Implement backward pass for a 2-layer network
# Given: forward pass values (z1, a1, z2, a2, loss)
# Compute: dW1, db1, dW2, db2
# Use: ReLU for hidden, softmax + cross-entropy for output
```

### Exercise B3 🟡 — Gradient Checking
```python
# TODO: Implement gradient checking for a neural network
# 1. Compute analytical gradients via backprop
# 2. Compute numerical gradients for each parameter
# 3. Compute relative difference: |analytical - numerical| / (|analytical| + |numerical|)
# 4. Flag if difference > 1e-5
```

### Exercise B4 🔴 — Build Autograd from Scratch
```python
# TODO: Implement a simple automatic differentiation system
# 1. Create a Tensor class that tracks computation graph
# 2. Implement forward ops: add, multiply, relu, sigmoid
# 3. Implement backward() that computes all gradients
# 4. Test on a simple computation
```

### Exercise B5 🔴 — Visualize Gradient Flow
```python
# TODO: Visualize gradient magnitude through layers
# 1. Create a deep network (10+ layers)
# 2. Forward and backward pass
# 3. Record gradient magnitude at each layer
# 4. Plot and compare: sigmoid vs ReLU activations
```

---

## Part C: Conceptual

### C1 🟡
Why is reverse-mode autodiff (backprop) more efficient than forward-mode for neural networks?

### C2 🟡
Explain why batch normalization helps with gradient flow. Connect to the entropy perspective.

### C3 🔴
If backprop computes exact gradients, why do we still have optimization difficulties in deep networks?

### C4 🔴
The cross-entropy loss connects backpropagation to information theory. Can you think of other loss functions with information-theoretic interpretations?
