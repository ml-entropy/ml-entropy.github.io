# Tutorial 03: Directional Derivatives - Solutions

## Part A: Theory Solutions

### Solution A1: Computing Directional Derivatives

**$f(x, y) = x^2 + xy + y^2$**

**1. Gradient:**
$$\nabla f = \begin{bmatrix} 2x + y \\ x + 2y \end{bmatrix}$$

**2. At $(1, 2)$ in direction $(3, 4)$:**
$$\nabla f(1, 2) = \begin{bmatrix} 2(1) + 2 \\ 1 + 2(2) \end{bmatrix} = \begin{bmatrix} 4 \\ 5 \end{bmatrix}$$

Normalize direction: $\mathbf{v} = \frac{1}{5}(3, 4)$

$$D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v} = \begin{bmatrix} 4 \\ 5 \end{bmatrix} \cdot \frac{1}{5}\begin{bmatrix} 3 \\ 4 \end{bmatrix} = \frac{1}{5}(12 + 20) = \boxed{\frac{32}{5} = 6.4}$$

**3. Direction of steepest ascent:**
$$\hat{\mathbf{v}} = \frac{\nabla f}{\|\nabla f\|} = \frac{1}{\sqrt{41}}\begin{bmatrix} 4 \\ 5 \end{bmatrix}$$

**4. Maximum rate of increase:**
$$\|\nabla f\| = \sqrt{16 + 25} = \boxed{\sqrt{41} \approx 6.4}$$

---

### Solution A2: Proof of Steepest Ascent

**1.** $D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v}$

**2.** Using $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$:
$$D_{\mathbf{v}} f = \|\nabla f\| \|\mathbf{v}\| \cos\theta$$

**3.** For unit vector $\|\mathbf{v}\| = 1$:
$$D_{\mathbf{v}} f = \|\nabla f\| \cos\theta$$

Maximized when $\cos\theta = 1$, i.e., $\theta = 0$.
This means $\mathbf{v}$ points in the same direction as $\nabla f$.

**4.** Maximum value:
$$\max_{\|\mathbf{v}\|=1} D_{\mathbf{v}} f = \|\nabla f\| \cdot 1 = \boxed{\|\nabla f\|}$$

---

### Solution A3: Gradient Perpendicular to Level Curves

**$f(x, y) = x^2 + 4y^2$**

**1. Level curve through $(2, 1)$:**
$$f(2, 1) = 4 + 4 = 8$$
Level curve: $x^2 + 4y^2 = 8$

**2. Tangent vector:**
Parameterize: $x = 2\sqrt{2}\cos t$, $y = \sqrt{2}\sin t$

At $(2, 1)$: $\cos t = \frac{1}{\sqrt{2}}$, $\sin t = \frac{1}{\sqrt{2}}$

Tangent: $\frac{d}{dt}(x, y) = (-2\sqrt{2}\sin t, \sqrt{2}\cos t) = (-2, 1)$

**3. Gradient:**
$$\nabla f = (2x, 8y) \Rightarrow \nabla f(2, 1) = (4, 8)$$

**4. Verify perpendicularity:**
$$\nabla f \cdot \text{tangent} = (4, 8) \cdot (-2, 1) = -8 + 8 = \boxed{0}$$

They are perpendicular! ✓

---

### Solution A4: Second Directional Derivative

**1. Derivation:**
$$D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v} = \sum_i \frac{\partial f}{\partial x_i} v_i$$

$$D_{\mathbf{v}}^2 f = D_{\mathbf{v}}(D_{\mathbf{v}} f) = \sum_i \frac{\partial}{\partial x_i}(D_{\mathbf{v}} f) \cdot v_i$$
$$= \sum_i \frac{\partial}{\partial x_i}\left(\sum_j \frac{\partial f}{\partial x_j} v_j\right) v_i$$
$$= \sum_{i,j} \frac{\partial^2 f}{\partial x_i \partial x_j} v_i v_j = \mathbf{v}^T H \mathbf{v}$$

**2. For $f(x, y) = x^2 + y^2$:**
$$H = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

$$D_{\mathbf{v}}^2 f = \frac{1}{2}\begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{2}(2 + 2) = \boxed{2}$$

**3. Interpretation:**
$D_{\mathbf{v}}^2 f > 0$ means the function curves upward in direction $\mathbf{v}$ (locally convex in that direction).

---

### Solution A5: Lagrange Multipliers

**1.** At optimum on constraint, any feasible direction along constraint shouldn't decrease $f$ (for minimum) or increase $f$ (for maximum).

**2.** For tangent vector $\mathbf{v}$ to constraint:
$$D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v} = 0$$

(Otherwise we could move along constraint to improve)

**3.** This means $\nabla f \perp \mathbf{v}$ for all tangent vectors $\mathbf{v}$.

Since $\nabla g \perp \mathbf{v}$ also (gradient perpendicular to level curves), both $\nabla f$ and $\nabla g$ are perpendicular to the constraint surface.

In 2D, this means $\nabla f$ and $\nabla g$ must be parallel:
$$\boxed{\nabla f = \lambda \nabla g}$$

---

## Part B: Coding Solutions

### Solution B1: Directional Derivative

```python
import numpy as np

def directional_derivative(f, x, direction, h=1e-5):
    x = np.array(x, dtype=float)
    v = np.array(direction, dtype=float)
    v = v / np.linalg.norm(v)  # Normalize
    
    return (f(x + h * v) - f(x - h * v)) / (2 * h)

# Test
f = lambda x: x[0]**2 + x[1]**2
point = np.array([1.0, 1.0])
direction = np.array([1.0, 1.0])

dd = directional_derivative(f, point, direction)
print(f"Directional derivative: {dd}")
print(f"Expected: 2*sqrt(2) = {2*np.sqrt(2)}")  # gradient = (2,2), dot with (1,1)/sqrt(2)
```

---

### Solution B2: Visualize Directional Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_directional_derivative(f, grad_f, point):
    angles = np.linspace(0, 2*np.pi, 100)
    grad = np.array(grad_f(*point))
    grad_angle = np.arctan2(grad[1], grad[0])
    grad_mag = np.linalg.norm(grad)
    
    dir_derivs = []
    for angle in angles:
        v = np.array([np.cos(angle), np.sin(angle)])
        dd = np.dot(grad, v)
        dir_derivs.append(dd)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cartesian
    axes[0].plot(np.degrees(angles), dir_derivs, 'b-', linewidth=2)
    axes[0].axhline(grad_mag, color='green', linestyle='--', label=f'Max = {grad_mag:.2f}')
    axes[0].axhline(-grad_mag, color='red', linestyle='--', label=f'Min = {-grad_mag:.2f}')
    axes[0].axvline(np.degrees(grad_angle), color='green', linestyle=':')
    axes[0].set_xlabel('Angle (degrees)')
    axes[0].set_ylabel('Directional derivative')
    axes[0].legend()
    axes[0].grid(True)
    
    # Polar
    ax = plt.subplot(122, projection='polar')
    ax.plot(angles, np.array(dir_derivs), 'b-')
    ax.plot([grad_angle], [grad_mag], 'go', markersize=10, label='Max')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Test
f = lambda x, y: x**2 + y**2
grad_f = lambda x, y: (2*x, 2*y)
visualize_directional_derivative(f, grad_f, (1, 2))
```

---

### Solution B3: Verify Perpendicularity

```python
import numpy as np

def verify_gradient_perpendicular(f, grad_f, level_value, num_points=20):
    # Sample points on level curve (for f = x² + y², level curves are circles)
    # f(x,y) = level_value means x² + y² = level_value
    r = np.sqrt(level_value)
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    
    dot_products = []
    for angle in angles:
        x, y = r * np.cos(angle), r * np.sin(angle)
        
        # Gradient at this point
        grad = np.array(grad_f(x, y))
        
        # Tangent to level curve (perpendicular to radial direction)
        tangent = np.array([-np.sin(angle), np.cos(angle)])
        
        # Dot product
        dot = np.dot(grad, tangent)
        dot_products.append(dot)
    
    print(f"Dot products (should be ≈ 0): {np.array(dot_products)}")
    print(f"Max absolute dot product: {np.max(np.abs(dot_products)):.2e}")
    return dot_products

# Test
f = lambda x, y: x**2 + y**2
grad_f = lambda x, y: (2*x, 2*y)
verify_gradient_perpendicular(f, grad_f, level_value=4)
```

---

### Solution B4: Compare Descent Methods

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_descent_methods(f, grad_f, start, n_steps=50, lr=0.1):
    results = {}
    
    # 1. Steepest descent
    point = np.array(start, dtype=float)
    traj_steepest = [point.copy()]
    for _ in range(n_steps):
        grad = np.array(grad_f(point[0], point[1]))
        point = point - lr * grad
        traj_steepest.append(point.copy())
    results['steepest'] = np.array(traj_steepest)
    
    # 2. Coordinate descent
    point = np.array(start, dtype=float)
    traj_coord = [point.copy()]
    for i in range(n_steps):
        grad = np.array(grad_f(point[0], point[1]))
        if i % 2 == 0:
            point[0] -= lr * grad[0]
        else:
            point[1] -= lr * grad[1]
        traj_coord.append(point.copy())
    results['coordinate'] = np.array(traj_coord)
    
    # 3. Random direction
    point = np.array(start, dtype=float)
    traj_random = [point.copy()]
    for _ in range(n_steps):
        direction = np.random.randn(2)
        direction = direction / np.linalg.norm(direction)
        grad = np.array(grad_f(point[0], point[1]))
        step = np.dot(grad, direction) * direction  # Project gradient onto random dir
        point = point - lr * step
        traj_random.append(point.copy())
    results['random'] = np.array(traj_random)
    
    # Plot
    x = np.linspace(-3, 3, 50)
    y = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    
    for name, traj in results.items():
        plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=3, label=name)
    
    plt.legend()
    plt.title('Comparison of Descent Methods')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    return results

# Test
f = lambda x, y: x**2 + 10*y**2
grad_f = lambda x, y: (2*x, 20*y)
compare_descent_methods(f, grad_f, [2.5, 2.0], n_steps=30, lr=0.05)
```

---

### Solution B5: Ball Rolling Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_ball_rolling(f, grad_f, start, friction=0.1, dt=0.01, n_steps=1000):
    x = np.array(start, dtype=float)
    v = np.zeros(2)  # Initial velocity
    
    trajectory = [x.copy()]
    velocities = [v.copy()]
    
    for _ in range(n_steps):
        # Acceleration = -gradient (downhill force) - friction * velocity
        grad = np.array(grad_f(x[0], x[1]))
        a = -grad - friction * v
        
        # Euler integration
        v = v + a * dt
        x = x + v * dt
        
        trajectory.append(x.copy())
        velocities.append(v.copy())
        
        # Stop if converged
        if np.linalg.norm(v) < 1e-6 and np.linalg.norm(grad) < 1e-6:
            break
    
    trajectory = np.array(trajectory)
    
    # Plot
    xg = np.linspace(-3, 3, 50)
    yg = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(xg, yg)
    Z = f(X, Y)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1, alpha=0.7)
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, label='Start')
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, label='End')
    plt.legend()
    plt.title('Ball Rolling on Surface (Physics-Based Gradient Descent)')
    plt.show()
    
    return trajectory, velocities

# Test
f = lambda x, y: x**2 + y**2
grad_f = lambda x, y: (2*x, 2*y)
simulate_ball_rolling(f, grad_f, [2.0, 1.5], friction=0.5)
```

---

## Part C: Conceptual Answers

### C1: Zero Directional Derivative
If $D_{\mathbf{v}} f = 0$, we're moving along a level curve (contour) of $f$. The function neither increases nor decreases in direction $\mathbf{v}$.

### C2: At Local Minimum
- All directional derivatives are ≥ 0 (can't decrease in any direction)
- The gradient $\nabla f = 0$ (otherwise some direction would have negative derivative)

### C3: Non-Gradient Descent
Yes, any direction with negative directional derivative will decrease $f$. However:
- Gradient direction gives the **fastest** decrease per unit step
- Other directions waste computation on smaller improvements
- Exception: momentum/Adam may use non-gradient directions for long-term benefit

### C4: Zigzagging
On elongated surfaces (different curvatures in different directions):
- Gradient points "toward" the minimum but at an angle to the direct path
- Each step overshoots the narrow direction, causing oscillation
- Momentum averages out oscillations, allowing faster progress along the valley

### C5: Natural Gradient
Standard gradient descent uses Euclidean distance in parameter space. But:
- Equal parameter changes may cause very different distribution changes
- Natural gradient accounts for the geometry of the distribution space
- $F^{-1} \nabla f$ moves equally in "distribution space" regardless of parameterization
- Often converges faster, especially for probabilistic models
