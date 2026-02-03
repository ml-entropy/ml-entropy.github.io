# Tutorial 08: Backpropagation — From First Principles to Entropy

## Overview
This tutorial derives backpropagation from scratch, building intuition for why it works and revealing its deep connection to information theory and entropy.

---

## Part 1: The Problem — How to Train Neural Networks

### What We Want
Given a neural network $f(x; \theta)$ with parameters $\theta$, find:
$$\theta^* = \arg\min_\theta \mathcal{L}(\theta)$$

where $\mathcal{L}$ is some loss function (e.g., cross-entropy, MSE).

### The Challenge
For a network with millions of parameters, we need:
$$\frac{\partial \mathcal{L}}{\partial \theta_i} \quad \text{for all } i$$

**Naive approach**: Compute each derivative numerically:
$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}$$

This requires **2 forward passes per parameter** — impossibly slow!

**Backpropagation solves this**: Compute ALL gradients in ONE backward pass.

---

## Part 2: The Chain Rule — Foundation of Backprop

### Single Variable Chain Rule
If $y = f(g(x))$, then:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

### Multivariable Chain Rule
If $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:
$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$$

### General Form (Total Derivative)
If $L$ depends on $\theta$ through intermediate variables $z_1, z_2, ..., z_n$:
$$\frac{dL}{d\theta} = \sum_{i=1}^{n} \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial \theta}$$

---

## Part 3: Computational Graphs

### Neural Network as a Graph
Every neural network is a **directed acyclic graph (DAG)**:
- **Nodes**: Operations (add, multiply, activation functions)
- **Edges**: Data flow (tensors)

### Example: Simple 2-Layer Network
```
Input x → [Linear W₁] → z₁ → [ReLU] → a₁ → [Linear W₂] → z₂ → [Softmax] → ŷ → [Loss] → L
```

Each arrow represents a function whose derivative we need.

### Forward Pass
Compute values left-to-right:
```
z₁ = W₁x + b₁
a₁ = ReLU(z₁)
z₂ = W₂a₁ + b₂
ŷ = softmax(z₂)
L = CrossEntropy(y, ŷ)
```

### Backward Pass
Compute gradients right-to-left using chain rule.

---

## Part 4: Deriving Backpropagation

### Step 1: Define the Computation

For a single neuron:
$$z = \sum_{i} w_i x_i + b = \mathbf{w}^T \mathbf{x} + b$$
$$a = \sigma(z)$$

where $\sigma$ is an activation function.

### Step 2: Loss Function Gradient

Start with loss $L$ and output $\hat{y}$.

For cross-entropy with softmax:
$$L = -\sum_k y_k \log \hat{y}_k$$

The gradient w.r.t. logits $z$ (before softmax) is remarkably simple:
$$\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$$

**Derivation**:
$$\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

$$\frac{\partial L}{\partial z_k} = -\sum_i y_i \frac{\partial \log \hat{y}_i}{\partial z_k}$$

For $i = k$:
$$\frac{\partial \log \hat{y}_k}{\partial z_k} = 1 - \hat{y}_k$$

For $i \neq k$:
$$\frac{\partial \log \hat{y}_i}{\partial z_k} = -\hat{y}_k$$

Combining:
$$\frac{\partial L}{\partial z_k} = -y_k(1 - \hat{y}_k) + \sum_{i \neq k} y_i \hat{y}_k = \hat{y}_k \sum_i y_i - y_k = \hat{y}_k - y_k$$

### Step 3: Propagate Through Layers

**Key insight**: Define $\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$ (error signal at layer $l$).

**Backward recurrence**:
$$\delta^{(l)} = \left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

where $\odot$ is element-wise multiplication.

**Weight gradients**:
$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^T$$

$$\frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

### Step 4: The Complete Algorithm

```
FORWARD PASS:
  for l = 1 to L:
    z[l] = W[l] @ a[l-1] + b[l]
    a[l] = activation(z[l])
  
  L = loss(a[L], y)

BACKWARD PASS:
  δ[L] = ∂L/∂z[L]  # Depends on loss function
  
  for l = L-1 down to 1:
    δ[l] = (W[l+1].T @ δ[l+1]) * activation_derivative(z[l])
    
  GRADIENTS:
  for l = 1 to L:
    ∂L/∂W[l] = δ[l] @ a[l-1].T
    ∂L/∂b[l] = δ[l]
```

---

## Part 5: Derivation for Specific Activations

### ReLU: $\sigma(z) = \max(0, z)$

$$\sigma'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$$

Backprop: gradient passes through if $z > 0$, else blocked.

### Sigmoid: $\sigma(z) = \frac{1}{1 + e^{-z}}$

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Derivation**:
$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(1 - \sigma)$$

### Tanh: $\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

$$\sigma'(z) = 1 - \tanh^2(z)$$

---

## Part 6: Matrix Form for Batched Computation

For a batch of $m$ samples, let $X \in \mathbb{R}^{n \times m}$ (each column is a sample).

**Forward**:
$$Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}$$
$$A^{(l)} = \sigma(Z^{(l)})$$

**Backward**:
$$dZ^{(l)} = dA^{(l)} \odot \sigma'(Z^{(l)})$$
$$dW^{(l)} = \frac{1}{m} dZ^{(l)} (A^{(l-1)})^T$$
$$db^{(l)} = \frac{1}{m} \sum_i dZ^{(l)}_i$$
$$dA^{(l-1)} = (W^{(l)})^T dZ^{(l)}$$

---

## Part 7: The Entropy Connection 🔗

### Why Cross-Entropy Loss?

The cross-entropy loss:
$$L = -\sum_k y_k \log \hat{y}_k = H(y, \hat{y})$$

is the **cross-entropy** between true distribution $y$ and predicted $\hat{y}$.

### Information-Theoretic View

$$H(y, \hat{y}) = H(y) + D_{KL}(y || \hat{y})$$

Since $y$ is fixed (ground truth), minimizing cross-entropy = minimizing KL divergence.

**Backpropagation minimizes the "surprise" of the network's predictions!**

### Why This Loss Works So Well

1. **Gradient is error**: $\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k$
   - Gradient IS the prediction error
   - No vanishing gradients from loss itself

2. **Probabilistic interpretation**: Network outputs a probability distribution
   - Maximum likelihood estimation
   - Principled uncertainty quantification

3. **Information efficiency**: Cross-entropy measures bits needed to encode $y$ using code designed for $\hat{y}$
   - Perfect prediction → minimum bits → zero loss

### The Deep Connection: Backprop as Information Flow

Consider what backpropagation really computes:

$$\delta^{(l)} = \frac{\partial L}{\partial z^{(l)}}$$

This is **how much the loss would change** if we changed $z^{(l)}$.

Information-theoretically:
- High $|\delta|$ = this neuron's output **matters a lot** for the prediction
- Low $|\delta|$ = this neuron is **irrelevant** to current prediction

**Backprop computes the "information relevance" of each neuron!**

### Entropy and Gradient Flow

For sigmoid/tanh activations:
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

This is maximized when $\sigma(z) = 0.5$ (maximum entropy/uncertainty).

**Neurons at maximum uncertainty have maximum gradient flow!**

This explains:
- **Vanishing gradients**: Saturated neurons (low entropy) block gradient flow
- **Dying ReLU**: Neurons stuck at 0 have zero entropy in their output
- **Batch normalization**: Keeps activations in high-gradient (high-entropy) regime

### The Fundamental Equation

Combining everything:

$$\boxed{\frac{\partial H(y, \hat{y})}{\partial \theta} = \text{Backprop computes how to reduce prediction surprise}}$$

**Machine learning IS entropy minimization through gradient flow.**

---

## Part 8: Automatic Differentiation

### Forward Mode vs Reverse Mode

**Forward mode**: Compute $\frac{\partial \text{output}}{\partial \text{one input}}$ efficiently
- Good when: few inputs, many outputs

**Reverse mode (backprop)**: Compute $\frac{\partial \text{one output}}{\partial \text{all inputs}}$ efficiently
- Good when: one output (scalar loss), many inputs (parameters)

Neural networks: ONE loss, MILLIONS of parameters → reverse mode wins!

### Computational Complexity

| Method | Forward Passes | Backward Passes | Total |
|--------|---------------|-----------------|-------|
| Numerical | $2n$ | 0 | $O(n)$ forward |
| Forward AD | $n$ | 0 | $O(n)$ forward |
| Reverse AD (backprop) | 1 | 1 | $O(1)$ forward + $O(1)$ backward |

For $n$ = millions of parameters, backprop is **millions of times faster**!

---

## Summary

1. **Backprop is repeated chain rule** applied to computational graphs
2. **Error signals propagate backward** through the network
3. **Gradients are local**: Each layer only needs its input, output, and upstream gradient
4. **Cross-entropy loss connects to information theory**: We minimize surprise
5. **Gradient magnitude ∝ information relevance**: Important neurons get larger gradients
6. **Reverse-mode AD** makes this computationally tractable

The beautiful insight: **Training neural networks is information-theoretic optimization** — we find parameters that minimize the surprise (entropy) of our predictions given the data.
