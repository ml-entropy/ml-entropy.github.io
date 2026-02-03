# Tutorial 02: Multivariable Derivatives - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Compute partial derivatives**

For the following functions, compute $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$:

1. $f(x, y) = x^2y + xy^2$
2. $f(x, y) = e^{xy}$
3. $f(x, y) = \ln(x^2 + y^2)$
4. $f(x, y) = \frac{x}{x + y}$

---

### Exercise A2 🟢 (Easy)
**Gradient vector**

For $f(x, y, z) = x^2 + 2y^2 + 3z^2 - xy$:

1. Compute the gradient $\nabla f$
2. Evaluate $\nabla f$ at point $(1, 1, 1)$
3. What is the magnitude $\|\nabla f\|$ at this point?
4. In what direction does $f$ increase most rapidly?

---

### Exercise A3 🟡 (Medium)
**Jacobian matrix**

For the transformation $\mathbf{F}: \mathbb{R}^2 \to \mathbb{R}^2$:
$$\mathbf{F}(x, y) = \begin{bmatrix} x^2 - y^2 \\ 2xy \end{bmatrix}$$

1. Compute the Jacobian matrix $J_F$
2. Evaluate $J_F$ at $(1, 1)$
3. What is $\det(J_F)$? What does it represent geometrically?

---

### Exercise A4 🟡 (Medium)
**Hessian matrix**

For $f(x, y) = x^3 - 3xy + y^3$:

1. Compute all second partial derivatives
2. Write the Hessian matrix $H$
3. Find the eigenvalues of $H$ at the point $(1, 1)$
4. Is this point a local min, max, or saddle point?

---

### Exercise A5 🔴 (Hard)
**Chain rule with multiple paths**

Let $z = f(u, v)$ where $u = x^2 + y$ and $v = xy$.

1. Derive $\frac{\partial z}{\partial x}$ in terms of $\frac{\partial f}{\partial u}$ and $\frac{\partial f}{\partial v}$
2. Derive $\frac{\partial z}{\partial y}$
3. If $f(u, v) = u^2 + v^2$, compute $\frac{\partial z}{\partial x}$ and $\frac{\partial z}{\partial y}$ explicitly

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Numerical gradient**

```python
def numerical_gradient_2d(f, x, y, h=1e-5):
    """
    Compute gradient of f(x, y) numerically.
    
    Returns: (df/dx, df/dy)
    """
    # YOUR CODE HERE
    pass

# Test on f(x,y) = x² + y². Gradient should be (2x, 2y).
```

---

### Exercise B2 🟡 (Medium)
**Gradient descent implementation**

```python
def gradient_descent_2d(f, grad_f, start, lr=0.1, n_steps=50):
    """
    Perform gradient descent on a 2D function.
    
    Args:
        f: Function to minimize
        grad_f: Gradient function returning (df/dx, df/dy)
        start: Starting point (x0, y0)
        lr: Learning rate
        n_steps: Number of iterations
    
    Returns:
        List of (x, y) points visited during optimization
    """
    # YOUR CODE HERE
    pass

# Test on Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
```

---

### Exercise B3 🟡 (Medium)
**Numerical Jacobian**

```python
def numerical_jacobian(F, x, h=1e-5):
    """
    Compute Jacobian matrix of F: R^n -> R^m numerically.
    
    Args:
        F: Vector function taking array of length n, returning array of length m
        x: Point at which to evaluate Jacobian (length n)
    
    Returns:
        Jacobian matrix of shape (m, n)
    """
    # YOUR CODE HERE
    pass

# Test on F(x,y) = (x²+y², x*y). J should be [[2x, 2y], [y, x]]
```

---

### Exercise B4 🔴 (Hard)
**Newton's method with Hessian**

```python
def newtons_method_2d(f, grad_f, hessian_f, start, n_steps=20):
    """
    Newton's method for optimization: x_{k+1} = x_k - H^{-1} * grad
    
    This uses second-order information (curvature) for faster convergence.
    
    Returns:
        trajectory: List of points visited
        converged: Boolean indicating if converged
    """
    # YOUR CODE HERE
    pass

# Test on f(x,y) = x² + 4y² (should converge very fast)
```

---

### Exercise B5 🔴 (Hard)
**Visualize gradient flow**

```python
def visualize_gradient_flow(f, grad_f, x_range, y_range, start_points):
    """
    Create visualization showing:
    1. Contour plot of f
    2. Gradient vector field
    3. Multiple gradient descent trajectories from different start points
    
    This helps build intuition about gradient-based optimization.
    """
    # YOUR CODE HERE
    pass

# Test on "two hills" function: f = -exp(-(x-1)²-y²) - exp(-(x+1)²-y²)
```

---

## Part C: Conceptual Questions

### C1 🟢
What's the difference between $\frac{\partial f}{\partial x}$ and $\frac{df}{dx}$? When are they the same?

### C2 🟢
Explain why the gradient points in the direction of steepest ascent. What would happen if we used a non-unit direction vector in the directional derivative formula?

### C3 🟡
For a neural network loss $L(\theta)$ with millions of parameters $\theta$:
- What is the dimension of $\nabla_\theta L$?
- What is the dimension of the Hessian $H_L$?
- Why do we typically only use gradients, not Hessians, in deep learning?

### C4 🟡
The Jacobian of a composition is the product of Jacobians: $J_{f \circ g} = J_f \cdot J_g$. How does this relate to the chain rule? Why does backpropagation multiply matrices in reverse order?

### C5 🔴
For a function $f: \mathbb{R}^n \to \mathbb{R}$, when is the Hessian guaranteed to be symmetric? Prove it using the definition of second partial derivatives.
