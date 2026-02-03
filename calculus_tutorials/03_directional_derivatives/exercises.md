# Tutorial 03: Directional Derivatives - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Compute directional derivatives**

For $f(x, y) = x^2 + xy + y^2$:

1. Compute the gradient $\nabla f$
2. Find the directional derivative at $(1, 2)$ in direction $\mathbf{v} = (3, 4)$
3. Find the direction of steepest ascent at $(1, 2)$
4. What is the maximum rate of increase?

---

### Exercise A2 🟡 (Medium)
**Prove the gradient is steepest ascent**

1. Write the directional derivative formula: $D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v}$
2. Use the formula $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$
3. Show that for unit vector $\mathbf{v}$, $D_{\mathbf{v}} f$ is maximized when $\mathbf{v}$ points in the same direction as $\nabla f$
4. What is this maximum value?

---

### Exercise A3 🟡 (Medium)
**Gradient perpendicular to level curves**

For $f(x, y) = x^2 + 4y^2$:

1. Write the equation of the level curve through $(2, 1)$
2. Find a tangent vector to this level curve at $(2, 1)$
3. Compute $\nabla f(2, 1)$
4. Verify that the gradient is perpendicular to the tangent vector

---

### Exercise A4 🔴 (Hard)
**Second directional derivative**

The second directional derivative is $D_{\mathbf{v}}^2 f = D_{\mathbf{v}}(D_{\mathbf{v}} f)$.

1. Show that $D_{\mathbf{v}}^2 f = \mathbf{v}^T H \mathbf{v}$ where $H$ is the Hessian
2. For $f(x, y) = x^2 + y^2$, compute $D_{\mathbf{v}}^2 f$ for $\mathbf{v} = \frac{1}{\sqrt{2}}(1, 1)$
3. What does $D_{\mathbf{v}}^2 f > 0$ tell us about the function in direction $\mathbf{v}$?

---

### Exercise A5 🔴 (Hard)
**Constrained optimization via directional derivatives**

Consider minimizing $f(x, y)$ subject to $g(x, y) = c$.

1. Explain why at an optimum, we cannot move along the constraint to decrease $f$
2. What does this say about $D_{\mathbf{v}} f$ where $\mathbf{v}$ is tangent to the constraint?
3. Show that this implies $\nabla f = \lambda \nabla g$ for some $\lambda$ (Lagrange multiplier condition)

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Compute directional derivative**

```python
def directional_derivative(f, x, direction, h=1e-5):
    """
    Compute directional derivative of f at point x in given direction.
    
    Args:
        f: Function R^n -> R
        x: Point as numpy array
        direction: Direction vector (will be normalized)
        h: Step size for numerical differentiation
    
    Returns:
        Directional derivative value
    """
    # YOUR CODE HERE
    pass

# Test: f(x,y) = x² + y² at (1, 1) in direction (1, 1)
# Should equal 2√2 ≈ 2.83
```

---

### Exercise B2 🟡 (Medium)
**Visualize directional derivatives**

```python
def visualize_directional_derivative(f, point, grad):
    """
    Create a polar plot showing directional derivative as function of angle.
    
    1. For angles 0 to 2π, compute directional derivative
    2. Plot in polar coordinates (angle, |D_v f|)
    3. Mark the gradient direction
    4. Show that max/min are along/opposite to gradient
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B3 🟡 (Medium)
**Verify gradient perpendicular to level curves**

```python
def verify_gradient_perpendicular(f, grad_f, level_value, num_points=20):
    """
    For a function f:
    1. Find points on level curve f(x,y) = level_value
    2. At each point, compute gradient and tangent to level curve
    3. Verify they are perpendicular (dot product ≈ 0)
    
    Return: List of dot products (should all be ≈ 0)
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B4 🔴 (Hard)
**Steepest descent vs coordinate descent**

```python
def compare_descent_methods(f, grad_f, start, n_steps=50, lr=0.1):
    """
    Compare three optimization strategies:
    1. Steepest descent (move along -gradient)
    2. Coordinate descent (alternate x and y directions)
    3. Random direction descent
    
    Return trajectories and final function values for each.
    """
    # YOUR CODE HERE
    pass

# Test on elongated function f(x,y) = x² + 10y²
```

---

### Exercise B5 🔴 (Hard)
**Gradient flow on a surface**

```python
def simulate_ball_rolling(f, grad_f, start, friction=0.1, dt=0.01, n_steps=1000):
    """
    Simulate a ball rolling on surface z = f(x, y) under gravity.
    
    Physics:
    - Ball accelerates in direction of -∇f (downhill)
    - Friction proportional to velocity
    - Use Euler integration
    
    This gives intuition for why gradient descent finds minima!
    
    Returns:
        trajectory: List of (x, y) positions
        velocities: List of (vx, vy) velocities
    """
    # YOUR CODE HERE
    pass
```

---

## Part C: Conceptual Questions

### C1 🟢
If $D_{\mathbf{v}} f = 0$ for some direction $\mathbf{v}$, what does this tell us geometrically?

### C2 🟢
At a local minimum, what must be true about all directional derivatives? What about the gradient?

### C3 🟡
In gradient descent, we move in direction $-\nabla f$. If we used a different direction (but still with negative directional derivative), would the function still decrease? Why might the gradient direction be optimal?

### C4 🟡
Explain why steepest descent can zigzag on elongated loss surfaces. How does momentum help?

### C5 🔴
In natural gradient descent, we move in direction $F^{-1} \nabla f$ where $F$ is the Fisher information matrix. Why might this be better than standard gradient descent? (Hint: Think about the geometry of probability distributions.)
