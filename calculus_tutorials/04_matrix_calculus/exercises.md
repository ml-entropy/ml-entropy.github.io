# Tutorial 04: Matrix Calculus - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Scalar-by-vector derivatives**

Derive the following:
1. $\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x})$ where $\mathbf{a}$ is constant
2. $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{x})$
3. $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{a})$ — is this different from (1)?

---

### Exercise A2 🟡 (Medium)
**Quadratic form derivative**

For $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ where $A$ is an $n \times n$ matrix:

1. Write out $f(\mathbf{x})$ using index notation: $f = \sum_{i,j} ...$
2. Compute $\frac{\partial f}{\partial x_k}$ by differentiating term by term
3. Show this equals $(A + A^T)\mathbf{x}$
4. If $A$ is symmetric, simplify the result

---

### Exercise A3 🟡 (Medium)
**Jacobian of linear transformation**

For $\mathbf{y} = A\mathbf{x}$ where $A$ is $m \times n$:

1. Write the Jacobian definition: $J_{ij} = \frac{\partial y_i}{\partial x_j}$
2. Compute $J_{ij}$ explicitly
3. Show that $J = A$

---

### Exercise A4 🔴 (Hard)
**Backprop through linear layer**

For a linear layer $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ with loss $\mathcal{L}$:

1. Given $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$, derive $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$
2. Derive $\frac{\partial \mathcal{L}}{\partial W}$
3. Derive $\frac{\partial \mathcal{L}}{\partial \mathbf{b}}$
4. Check dimensions match

---

### Exercise A5 🔴 (Hard)
**Softmax Jacobian**

For softmax: $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$

1. Compute $\frac{\partial p_i}{\partial z_j}$ for $i = j$
2. Compute $\frac{\partial p_i}{\partial z_j}$ for $i \neq j$
3. Write the full Jacobian in matrix form
4. Verify that rows sum to zero (why must this be true?)

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Verify matrix calculus identities**

```python
def verify_identity_1():
    """
    Verify: ∂/∂x(a^T x) = a
    
    Use numerical gradient to check.
    """
    # YOUR CODE HERE
    pass

def verify_identity_2():
    """
    Verify: ∂/∂x(x^T A x) = (A + A^T)x
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B2 🟡 (Medium)
**Implement linear layer backprop**

```python
class LinearLayer:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.01
        self.b = np.zeros(out_features)
        
    def forward(self, x):
        """y = Wx + b"""
        # YOUR CODE HERE (also cache x for backward)
        pass
    
    def backward(self, grad_y):
        """
        Given dL/dy, compute:
        - dL/dx
        - dL/dW
        - dL/db
        """
        # YOUR CODE HERE
        pass

# Verify with numerical gradients
```

---

### Exercise B3 🟡 (Medium)
**Implement softmax with backprop**

```python
class Softmax:
    def forward(self, z):
        """p = softmax(z)"""
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_p):
        """
        Given dL/dp, compute dL/dz
        
        Use the Jacobian: dL/dz = J^T @ dL/dp
        """
        # YOUR CODE HERE
        pass

# Test: softmax + cross-entropy should give gradient (p - y)
```

---

### Exercise B4 🔴 (Hard)
**Two-layer network with full backprop**

```python
class TwoLayerNetwork:
    """
    y = W2 @ relu(W1 @ x + b1) + b2
    """
    def __init__(self, d_in, d_hidden, d_out):
        # YOUR CODE HERE
        pass
    
    def forward(self, x):
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_loss):
        """
        Compute gradients for all parameters using chain rule.
        """
        # YOUR CODE HERE
        pass
    
    def numerical_gradient(self, x, loss_fn):
        """For verification: compute gradients numerically."""
        # YOUR CODE HERE
        pass
```

---

### Exercise B5 🔴 (Hard)
**Batch matrix operations**

```python
def batch_linear_backward(X, W, grad_Y):
    """
    Batch version of linear layer backward.
    
    Forward: Y = X @ W^T (batch_size × out_features)
    X: (batch_size, in_features)
    W: (out_features, in_features)
    grad_Y: (batch_size, out_features)
    
    Compute:
    - grad_X: (batch_size, in_features)
    - grad_W: (out_features, in_features) — summed over batch
    """
    # YOUR CODE HERE
    pass
```

---

## Part C: Conceptual Questions

### C1 🟢
Why is $\frac{\partial}{\partial \mathbf{x}}(A\mathbf{x}) = A$ but $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A) = A^T$?

### C2 🟢
In backprop through a linear layer $\mathbf{y} = W\mathbf{x}$, why do we multiply by $W^T$ when computing $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$?

### C3 🟡
For a composition $\mathbf{z} = f(\mathbf{y})$ and $\mathbf{y} = g(\mathbf{x})$, show that $J_{\mathbf{z}/\mathbf{x}} = J_{f} \cdot J_{g}$. Why does backprop multiply in reverse order?

### C4 🟡
The gradient of a scalar w.r.t. a matrix has the same shape as the matrix. Why is this useful for gradient descent?

### C5 🔴
In deep learning, we often use the "numerator layout" convention. What changes if we use "denominator layout"? How do the backprop equations change?
