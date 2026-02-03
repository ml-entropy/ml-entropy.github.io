# Tutorial 04: Matrix Calculus - Solutions

## Part A: Theory Solutions

### Solution A1: Scalar-by-Vector Derivatives

**1. $\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x})$:**

$$\mathbf{a}^T \mathbf{x} = \sum_{i=1}^{n} a_i x_i$$

$$\frac{\partial}{\partial x_j}\left(\sum_i a_i x_i\right) = a_j$$

Therefore: $\boxed{\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}}$

**2. $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{x})$:**

$$\mathbf{x}^T \mathbf{x} = \sum_{i=1}^{n} x_i^2$$

$$\frac{\partial}{\partial x_j}\left(\sum_i x_i^2\right) = 2x_j$$

Therefore: $\boxed{\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{x}) = 2\mathbf{x}}$

**3. $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{a})$:**

$$\mathbf{x}^T \mathbf{a} = \sum_i x_i a_i = \mathbf{a}^T \mathbf{x}$$

Same as (1): $\boxed{\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{a}) = \mathbf{a}}$

---

### Solution A2: Quadratic Form Derivative

**1. Index notation:**
$$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i=1}^{n} \sum_{j=1}^{n} x_i A_{ij} x_j$$

**2. Differentiate:**
$$\frac{\partial f}{\partial x_k} = \frac{\partial}{\partial x_k} \sum_{i,j} x_i A_{ij} x_j$$

Terms where $i = k$: $\sum_j A_{kj} x_j = (A\mathbf{x})_k$
Terms where $j = k$: $\sum_i x_i A_{ik} = (A^T\mathbf{x})_k$

$$\frac{\partial f}{\partial x_k} = (A\mathbf{x})_k + (A^T\mathbf{x})_k = ((A + A^T)\mathbf{x})_k$$

**3. Result:**
$$\boxed{\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}}$$

**4. If $A$ is symmetric:** $A = A^T$, so:
$$\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x}) = 2A\mathbf{x}$$

---

### Solution A3: Jacobian of Linear Transformation

**1. Definition:**
$$J_{ij} = \frac{\partial y_i}{\partial x_j}$$

**2. Compute:**
$$y_i = (A\mathbf{x})_i = \sum_{k=1}^{n} A_{ik} x_k$$

$$\frac{\partial y_i}{\partial x_j} = A_{ij}$$

**3. Result:**
$$\boxed{J = A}$$

The Jacobian of a linear transformation is the transformation matrix itself.

---

### Solution A4: Backprop Through Linear Layer

**Given:** $\mathbf{y} = W\mathbf{x} + \mathbf{b}$, upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$

**1. Gradient w.r.t. input:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = W^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}}$

**2. Gradient w.r.t. weights:**
$$\frac{\partial \mathcal{L}}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\partial y_i}{\partial W_{ij}} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot x_j$$

$\boxed{\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^T}$ (outer product)

**3. Gradient w.r.t. bias:**
$$\frac{\partial y_i}{\partial b_i} = 1$$

$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}}$

**4. Dimension check:**
- $\mathbf{x}$: $(n,)$, $\mathbf{y}$: $(m,)$, $W$: $(m, n)$
- $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$: $(n,) = (n, m) \times (m,)$ ✓
- $\frac{\partial \mathcal{L}}{\partial W}$: $(m, n) = (m,) \times (n,)^T$ ✓
- $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}$: $(m,)$ ✓

---

### Solution A5: Softmax Jacobian

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**1. Case $i = j$:**
$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i} \cdot \sum_j e^{z_j} - e^{z_i} \cdot e^{z_i}}{(\sum_j e^{z_j})^2} = p_i - p_i^2 = \boxed{p_i(1 - p_i)}$$

**2. Case $i \neq j$:**
$$\frac{\partial p_i}{\partial z_j} = \frac{0 - e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = -p_i p_j = \boxed{-p_i p_j}$$

**3. Matrix form:**
$$J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

**4. Row sum:**
$$\sum_j J_{ij} = p_i(1 - p_i) + \sum_{j \neq i} (-p_i p_j) = p_i - p_i \sum_j p_j = p_i - p_i = 0$$

Rows sum to zero because probabilities sum to 1: if $z_j$ increases, some $p_i$ must decrease.

---

## Part B: Coding Solutions

### Solution B1: Verify Identities

```python
import numpy as np

def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def verify_identity_1():
    a = np.array([1., 2., 3.])
    x = np.array([0.5, 1.5, 2.5])
    
    f = lambda x: np.dot(a, x)
    numerical = numerical_gradient(f, x)
    analytical = a
    
    print("Identity 1: ∂/∂x(a^T x) = a")
    print(f"  Numerical: {numerical}")
    print(f"  Analytical: {analytical}")
    print(f"  Match: {np.allclose(numerical, analytical)}")

def verify_identity_2():
    A = np.random.randn(3, 3)
    x = np.array([1., 2., 3.])
    
    f = lambda x: x @ A @ x
    numerical = numerical_gradient(f, x)
    analytical = (A + A.T) @ x
    
    print("\nIdentity 2: ∂/∂x(x^T A x) = (A + A^T)x")
    print(f"  Numerical: {numerical}")
    print(f"  Analytical: {analytical}")
    print(f"  Match: {np.allclose(numerical, analytical)}")

verify_identity_1()
verify_identity_2()
```

---

### Solution B2: Linear Layer Backprop

```python
import numpy as np

class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros(out_features)
        self.x = None
        
    def forward(self, x):
        self.x = x
        return self.W @ x + self.b
    
    def backward(self, grad_y):
        grad_x = self.W.T @ grad_y
        grad_W = np.outer(grad_y, self.x)
        grad_b = grad_y
        return grad_x, grad_W, grad_b

# Verify
layer = LinearLayer(3, 2)
x = np.array([1., 2., 3.])
y = layer.forward(x)

# For L = sum(y), grad_y = [1, 1]
grad_y = np.ones(2)
grad_x, grad_W, grad_b = layer.backward(grad_y)

# Numerical verification
eps = 1e-5
numerical_grad_W = np.zeros_like(layer.W)
for i in range(layer.W.shape[0]):
    for j in range(layer.W.shape[1]):
        layer.W[i, j] += eps
        y_plus = layer.forward(x)
        layer.W[i, j] -= 2 * eps
        y_minus = layer.forward(x)
        layer.W[i, j] += eps
        numerical_grad_W[i, j] = (y_plus.sum() - y_minus.sum()) / (2 * eps)

print("Linear Layer Backprop Verification:")
print(f"  Analytical grad_W:\n{grad_W}")
print(f"  Numerical grad_W:\n{numerical_grad_W}")
print(f"  Match: {np.allclose(grad_W, numerical_grad_W)}")
```

---

### Solution B3: Softmax Backprop

```python
import numpy as np

class Softmax:
    def forward(self, z):
        exp_z = np.exp(z - np.max(z))  # Numerical stability
        self.p = exp_z / exp_z.sum()
        return self.p
    
    def backward(self, grad_p):
        # Jacobian: J = diag(p) - p @ p^T
        J = np.diag(self.p) - np.outer(self.p, self.p)
        return J.T @ grad_p

# Test: softmax + cross-entropy
# For cross-entropy loss L = -sum(y * log(p)), dL/dp = -y/p
# Combined gradient dL/dz should equal (p - y)

z = np.array([2.0, 1.0, 0.1])
y = np.array([1., 0., 0.])  # One-hot target

softmax = Softmax()
p = softmax.forward(z)

# Cross-entropy gradient w.r.t. p
grad_p = -y / (p + 1e-10)

# Softmax backward
grad_z = softmax.backward(grad_p)

print("Softmax + Cross-Entropy Gradient:")
print(f"  p = {p}")
print(f"  Computed grad_z = {grad_z}")
print(f"  Expected (p - y) = {p - y}")
print(f"  Match: {np.allclose(grad_z, p - y)}")
```

---

### Solution B4: Two-Layer Network

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class TwoLayerNetwork:
    def __init__(self, d_in, d_hidden, d_out):
        self.W1 = np.random.randn(d_hidden, d_in) * 0.1
        self.b1 = np.zeros(d_hidden)
        self.W2 = np.random.randn(d_out, d_hidden) * 0.1
        self.b2 = np.zeros(d_out)
    
    def forward(self, x):
        self.x = x
        self.z1 = self.W1 @ x + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        return self.z2
    
    def backward(self, grad_loss):
        # Backprop through output layer
        grad_W2 = np.outer(grad_loss, self.a1)
        grad_b2 = grad_loss
        grad_a1 = self.W2.T @ grad_loss
        
        # Backprop through ReLU
        grad_z1 = grad_a1 * relu_derivative(self.z1)
        
        # Backprop through hidden layer
        grad_W1 = np.outer(grad_z1, self.x)
        grad_b1 = grad_z1
        grad_x = self.W1.T @ grad_z1
        
        return {'W1': grad_W1, 'b1': grad_b1, 'W2': grad_W2, 'b2': grad_b2, 'x': grad_x}
    
    def numerical_gradient(self, x, loss_fn, param_name):
        eps = 1e-5
        param = getattr(self, param_name)
        grad = np.zeros_like(param)
        
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_val = param[idx]
            
            param[idx] = old_val + eps
            loss_plus = loss_fn(self.forward(x))
            
            param[idx] = old_val - eps
            loss_minus = loss_fn(self.forward(x))
            
            param[idx] = old_val
            grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        
        return grad

# Test
net = TwoLayerNetwork(4, 3, 2)
x = np.random.randn(4)
loss_fn = lambda y: np.sum(y)  # Simple loss for testing

y = net.forward(x)
grads = net.backward(np.ones(2))  # dL/dy = [1, 1]

# Verify W1 gradient
numerical_W1 = net.numerical_gradient(x, loss_fn, 'W1')
print("Two-Layer Network Gradient Check:")
print(f"  W1 gradient match: {np.allclose(grads['W1'], numerical_W1)}")
print(f"  Max diff: {np.max(np.abs(grads['W1'] - numerical_W1)):.2e}")
```

---

### Solution B5: Batch Operations

```python
import numpy as np

def batch_linear_backward(X, W, grad_Y):
    """
    X: (batch_size, in_features)
    W: (out_features, in_features)
    grad_Y: (batch_size, out_features)
    
    Forward: Y = X @ W^T
    """
    # grad_X: (batch_size, in_features)
    grad_X = grad_Y @ W
    
    # grad_W: (out_features, in_features) - sum over batch
    grad_W = grad_Y.T @ X
    
    return grad_X, grad_W

# Test
batch_size = 4
in_features = 3
out_features = 2

X = np.random.randn(batch_size, in_features)
W = np.random.randn(out_features, in_features)
Y = X @ W.T  # Forward pass

# Suppose loss = sum of all Y elements
grad_Y = np.ones((batch_size, out_features))
grad_X, grad_W = batch_linear_backward(X, W, grad_Y)

print("Batch Linear Backward:")
print(f"  grad_X shape: {grad_X.shape} (expected: {(batch_size, in_features)})")
print(f"  grad_W shape: {grad_W.shape} (expected: {(out_features, in_features)})")
```

---

## Part C: Conceptual Answers

### C1: $A\mathbf{x}$ vs $\mathbf{x}^T A$
- $A\mathbf{x}$: Each row of $A$ dots with $\mathbf{x}$. Jacobian is $A$.
- $\mathbf{x}^T A$: Each column of $A$ dots with $\mathbf{x}$. Result is a row vector, gradient is $A^T$.

### C2: Why $W^T$?
The Jacobian of $\mathbf{y} = W\mathbf{x}$ is $W$. To propagate gradients backward:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = J^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}} = W^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$

We transpose because we're going backward through the computation graph.

### C3: Jacobian of Composition
By chain rule: $\frac{\partial z_i}{\partial x_k} = \sum_j \frac{\partial z_i}{\partial y_j} \frac{\partial y_j}{\partial x_k}$

This is matrix multiplication: $J_{\mathbf{z}/\mathbf{x}} = J_f \cdot J_g$

Backprop multiplies in reverse because we compute $\nabla_{\mathbf{x}} \mathcal{L} = J^T \nabla_{\mathbf{y}} \mathcal{L}$, propagating from output to input.

### C4: Same Shape
If $\nabla_W \mathcal{L}$ has the same shape as $W$, we can update directly:
$$W_{new} = W_{old} - \eta \nabla_W \mathcal{L}$$

No reshaping needed!

### C5: Numerator vs Denominator Layout
- **Numerator layout**: Gradient is column vector, Jacobian has output rows
- **Denominator layout**: Gradient is row vector, Jacobian has input rows

In denominator layout:
- $\nabla_{\mathbf{x}} \mathcal{L} = \nabla_{\mathbf{y}} \mathcal{L} \cdot J$ (instead of $J^T \cdot \nabla$)
- Dimensions swap

Most ML frameworks use numerator layout. Always check which convention is used!
