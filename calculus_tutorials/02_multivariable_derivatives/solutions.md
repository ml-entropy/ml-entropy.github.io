# Tutorial 02: Multivariable Derivatives - Solutions

## Part A: Theory Solutions

### Solution A1: Partial Derivatives

**1. $f(x, y) = x^2y + xy^2$**
$$\frac{\partial f}{\partial x} = 2xy + y^2$$
$$\frac{\partial f}{\partial y} = x^2 + 2xy$$

**2. $f(x, y) = e^{xy}$**
$$\frac{\partial f}{\partial x} = ye^{xy}$$
$$\frac{\partial f}{\partial y} = xe^{xy}$$

**3. $f(x, y) = \ln(x^2 + y^2)$**
$$\frac{\partial f}{\partial x} = \frac{2x}{x^2 + y^2}$$
$$\frac{\partial f}{\partial y} = \frac{2y}{x^2 + y^2}$$

**4. $f(x, y) = \frac{x}{x + y}$**
$$\frac{\partial f}{\partial x} = \frac{(x+y) - x}{(x+y)^2} = \frac{y}{(x+y)^2}$$
$$\frac{\partial f}{\partial y} = \frac{0 - x}{(x+y)^2} = \frac{-x}{(x+y)^2}$$

---

### Solution A2: Gradient Vector

**$f(x, y, z) = x^2 + 2y^2 + 3z^2 - xy$**

**1. Gradient:**
$$\nabla f = \begin{bmatrix} 2x - y \\ 4y - x \\ 6z \end{bmatrix}$$

**2. At $(1, 1, 1)$:**
$$\nabla f(1,1,1) = \begin{bmatrix} 2(1) - 1 \\ 4(1) - 1 \\ 6(1) \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \\ 6 \end{bmatrix}$$

**3. Magnitude:**
$$\|\nabla f\| = \sqrt{1^2 + 3^2 + 6^2} = \sqrt{1 + 9 + 36} = \sqrt{46} \approx 6.78$$

**4. Direction of fastest increase:**
$$\hat{v} = \frac{\nabla f}{\|\nabla f\|} = \frac{1}{\sqrt{46}}\begin{bmatrix} 1 \\ 3 \\ 6 \end{bmatrix}$$

---

### Solution A3: Jacobian Matrix

**$\mathbf{F}(x, y) = \begin{bmatrix} x^2 - y^2 \\ 2xy \end{bmatrix}$**

**1. Jacobian:**
$$J_F = \begin{bmatrix} \frac{\partial(x^2-y^2)}{\partial x} & \frac{\partial(x^2-y^2)}{\partial y} \\ \frac{\partial(2xy)}{\partial x} & \frac{\partial(2xy)}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x & -2y \\ 2y & 2x \end{bmatrix}$$

**2. At $(1, 1)$:**
$$J_F(1,1) = \begin{bmatrix} 2 & -2 \\ 2 & 2 \end{bmatrix}$$

**3. Determinant:**
$$\det(J_F) = (2x)(2x) - (-2y)(2y) = 4x^2 + 4y^2 = 4(x^2 + y^2)$$

At $(1,1)$: $\det(J_F) = 8$

Geometric meaning: The determinant represents the local scaling factor of area. This transformation stretches areas by a factor of 8 near $(1,1)$.

---

### Solution A4: Hessian Matrix

**$f(x, y) = x^3 - 3xy + y^3$**

**1. First derivatives:**
$$\frac{\partial f}{\partial x} = 3x^2 - 3y, \quad \frac{\partial f}{\partial y} = -3x + 3y^2$$

**Second derivatives:**
$$\frac{\partial^2 f}{\partial x^2} = 6x, \quad \frac{\partial^2 f}{\partial y^2} = 6y, \quad \frac{\partial^2 f}{\partial x \partial y} = -3$$

**2. Hessian:**
$$H = \begin{bmatrix} 6x & -3 \\ -3 & 6y \end{bmatrix}$$

**3. At $(1, 1)$:**
$$H(1,1) = \begin{bmatrix} 6 & -3 \\ -3 & 6 \end{bmatrix}$$

Eigenvalues: $\det(H - \lambda I) = (6-\lambda)^2 - 9 = 0$
$(6-\lambda)^2 = 9 \Rightarrow 6-\lambda = \pm 3$
$\lambda_1 = 3, \lambda_2 = 9$

**4. Classification:**
Both eigenvalues are positive → $H$ is positive definite → **local minimum**

---

### Solution A5: Chain Rule with Multiple Paths

**$z = f(u, v)$ where $u = x^2 + y$, $v = xy$**

**1. $\frac{\partial z}{\partial x}$:**
$$\frac{\partial z}{\partial x} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial x} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial x}$$
$$= \frac{\partial f}{\partial u} \cdot 2x + \frac{\partial f}{\partial v} \cdot y$$

**2. $\frac{\partial z}{\partial y}$:**
$$\frac{\partial z}{\partial y} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial y} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial y}$$
$$= \frac{\partial f}{\partial u} \cdot 1 + \frac{\partial f}{\partial v} \cdot x$$

**3. For $f(u, v) = u^2 + v^2$:**
$\frac{\partial f}{\partial u} = 2u = 2(x^2+y)$, $\frac{\partial f}{\partial v} = 2v = 2xy$

$$\frac{\partial z}{\partial x} = 2(x^2+y) \cdot 2x + 2xy \cdot y = 4x^3 + 4xy + 2xy^2$$
$$\frac{\partial z}{\partial y} = 2(x^2+y) \cdot 1 + 2xy \cdot x = 2x^2 + 2y + 2x^2y$$

---

## Part B: Coding Solutions

### Solution B1: Numerical Gradient

```python
import numpy as np

def numerical_gradient_2d(f, x, y, h=1e-5):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return df_dx, df_dy

# Test
f = lambda x, y: x**2 + y**2
x, y = 3.0, 4.0
grad = numerical_gradient_2d(f, x, y)
print(f"Numerical gradient at ({x}, {y}): {grad}")
print(f"Analytical gradient: ({2*x}, {2*y})")
```

---

### Solution B2: Gradient Descent

```python
import numpy as np

def gradient_descent_2d(f, grad_f, start, lr=0.1, n_steps=50):
    trajectory = [np.array(start)]
    point = np.array(start, dtype=float)
    
    for _ in range(n_steps):
        grad = np.array(grad_f(point[0], point[1]))
        point = point - lr * grad
        trajectory.append(point.copy())
    
    return trajectory

# Test on Rosenbrock
def rosenbrock(x, y):
    return (1 - x)**2 + 100*(y - x**2)**2

def rosenbrock_grad(x, y):
    df_dx = -2*(1 - x) - 400*x*(y - x**2)
    df_dy = 200*(y - x**2)
    return [df_dx, df_dy]

trajectory = gradient_descent_2d(rosenbrock, rosenbrock_grad, [-1.0, 1.0], lr=0.001, n_steps=1000)
final = trajectory[-1]
print(f"Final point: {final}")
print(f"Optimal: [1, 1], f(1,1) = 0")
```

---

### Solution B3: Numerical Jacobian

```python
import numpy as np

def numerical_jacobian(F, x, h=1e-5):
    x = np.array(x, dtype=float)
    n = len(x)
    f0 = np.array(F(x))
    m = len(f0)
    
    J = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (np.array(F(x_plus)) - np.array(F(x_minus))) / (2 * h)
    
    return J

# Test
def F(x):
    return [x[0]**2 + x[1]**2, x[0] * x[1]]

x = np.array([2.0, 3.0])
J = numerical_jacobian(F, x)
print(f"Numerical Jacobian:\n{J}")
print(f"Analytical: [[2x, 2y], [y, x]] = [[4, 6], [3, 2]]")
```

---

### Solution B4: Newton's Method

```python
import numpy as np

def newtons_method_2d(f, grad_f, hessian_f, start, n_steps=20, tol=1e-8):
    trajectory = [np.array(start)]
    point = np.array(start, dtype=float)
    converged = False
    
    for _ in range(n_steps):
        grad = np.array(grad_f(point[0], point[1]))
        H = np.array(hessian_f(point[0], point[1]))
        
        # Newton step: x = x - H^(-1) @ grad
        try:
            step = np.linalg.solve(H, grad)
            point = point - step
            trajectory.append(point.copy())
            
            if np.linalg.norm(step) < tol:
                converged = True
                break
        except np.linalg.LinAlgError:
            break
    
    return trajectory, converged

# Test on f(x,y) = x² + 4y²
def f(x, y):
    return x**2 + 4*y**2

def grad_f(x, y):
    return [2*x, 8*y]

def hessian_f(x, y):
    return [[2, 0], [0, 8]]

traj, conv = newtons_method_2d(f, grad_f, hessian_f, [3.0, 2.0], n_steps=10)
print(f"Converged: {conv}")
print(f"Steps: {len(traj) - 1}")
print(f"Final point: {traj[-1]}")  # Should be [0, 0]
```

---

### Solution B5: Gradient Flow Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_gradient_flow(f, grad_f, x_range, y_range, start_points):
    x = np.linspace(*x_range, 50)
    y = np.linspace(*y_range, 50)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Gradient field
    U, V = [], []
    for xi in x:
        row_u, row_v = [], []
        for yi in y:
            gx, gy = grad_f(xi, yi)
            row_u.append(gx)
            row_v.append(gy)
        U.append(row_u)
        V.append(row_v)
    U, V = np.array(U).T, np.array(V).T
    
    plt.figure(figsize=(12, 8))
    
    # Contours
    plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    
    # Gradient field (subsample)
    skip = 3
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
               -U[::skip, ::skip], -V[::skip, ::skip],
               color='gray', alpha=0.5, scale=30)
    
    # Trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(start_points)))
    for start, color in zip(start_points, colors):
        traj = [np.array(start)]
        point = np.array(start, dtype=float)
        for _ in range(100):
            grad = np.array(grad_f(point[0], point[1]))
            point = point - 0.1 * grad
            traj.append(point.copy())
            if np.linalg.norm(grad) < 0.01:
                break
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], 'o-', color=color, markersize=3)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Flow')
    plt.show()

# Test
f = lambda x, y: -np.exp(-(x-1)**2 - y**2) - np.exp(-(x+1)**2 - y**2)
grad_f = lambda x, y: (2*(x-1)*np.exp(-(x-1)**2-y**2) + 2*(x+1)*np.exp(-(x+1)**2-y**2),
                       2*y*np.exp(-(x-1)**2-y**2) + 2*y*np.exp(-(x+1)**2-y**2))

visualize_gradient_flow(f, grad_f, [-3, 3], [-2, 2], 
                       [[-2, 1], [0, 1.5], [2, 0.5], [0, -1]])
```

---

## Part C: Conceptual Answers

### C1: Partial vs Total Derivative
- **Partial derivative** $\frac{\partial f}{\partial x}$: Rate of change when only $x$ varies, others held constant
- **Total derivative** $\frac{df}{dx}$: Rate of change accounting for all dependencies

They're the same when $f$ only depends on $x$ directly, not through other variables.

### C2: Gradient and Steepest Ascent
The directional derivative is $D_v f = \nabla f \cdot v = \|\nabla f\| \|v\| \cos\theta$.

For unit $v$ ($\|v\| = 1$), this is maximized when $\theta = 0$ (same direction as $\nabla f$).

For non-unit $v$, the rate of change scales with $\|v\|$.

### C3: Gradient vs Hessian in Deep Learning
- Gradient dimension: $n$ (same as number of parameters)
- Hessian dimension: $n \times n$

For $n = 10^8$ parameters: Hessian has $10^{16}$ entries!
- Too large to store or compute
- Gradient: $O(n)$ space, one backprop pass
- Hessian: $O(n^2)$ space, $O(n)$ backprop passes

### C4: Jacobian Product and Backprop
$J_{f \circ g} = J_f \cdot J_g$ is the matrix form of chain rule.

For $L = f_n \circ f_{n-1} \circ ... \circ f_1$:
$$J_L = J_{f_n} \cdot J_{f_{n-1}} \cdot ... \cdot J_{f_1}$$

We multiply right-to-left (reverse order) because we're computing $\frac{\partial L}{\partial x}$, starting from $\frac{\partial L}{\partial y_n} = 1$ and propagating backward.

### C5: Hessian Symmetry
By Schwarz's theorem, if $f$ has continuous second partial derivatives:
$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

So $H_{ij} = H_{ji}$ for all $i, j$, making $H$ symmetric.

Proof sketch: Both mixed partials equal $\lim_{h,k \to 0} \frac{f(x+h, y+k) - f(x+h, y) - f(x, y+k) + f(x, y)}{hk}$
