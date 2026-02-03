# Tutorial 02: Multivariable Derivatives

## From One Variable to Many

In ML, our loss functions depend on **thousands or millions of parameters**. We need to understand how the loss changes when we change any single parameter — while keeping others fixed.

---

## Partial Derivatives

### Definition

For a function $f(x_1, x_2, \ldots, x_n)$, the **partial derivative** with respect to $x_i$ is:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

**Key idea:** Treat all other variables as constants and differentiate normally.

### Example 1: Simple function

$f(x, y) = x^2 + 3xy + y^2$

**Partial w.r.t. x** (treat $y$ as constant):
$$\frac{\partial f}{\partial x} = 2x + 3y$$

**Partial w.r.t. y** (treat $x$ as constant):
$$\frac{\partial f}{\partial y} = 3x + 2y$$

### Example 2: Neural network layer

Consider a single neuron: $z = w_1 x_1 + w_2 x_2 + b$

**Partial derivatives:**
$$\frac{\partial z}{\partial w_1} = x_1, \quad \frac{\partial z}{\partial w_2} = x_2, \quad \frac{\partial z}{\partial b} = 1$$

These are exactly what we need for gradient descent!

---

## The Gradient Vector

### Definition

For $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** is the vector of all partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Geometric Interpretation

**The gradient points in the direction of steepest ascent.**

To minimize a function, we move **opposite** to the gradient:
$$\theta_{new} = \theta_{old} - \eta \nabla_\theta \mathcal{L}$$

### Example

$f(x, y) = x^2 + y^2$ (paraboloid bowl)

$$\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}$$

At point $(1, 2)$:
$$\nabla f(1, 2) = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$

The gradient points away from the minimum at origin. To minimize, move in direction $(-2, -4)$.

---

## The Jacobian Matrix

### When Output is a Vector

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ where $\mathbf{f}(\mathbf{x}) = [f_1(\mathbf{x}), \ldots, f_m(\mathbf{x})]^T$:

$$J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

**Entry $(i, j)$:** How much does output $i$ change when input $j$ changes?

### Example: Linear transformation

$\mathbf{f}(\mathbf{x}) = A\mathbf{x}$ where $A$ is $m \times n$ matrix.

The Jacobian is simply $J = A$!

This is why we multiply by weight matrices in backprop.

### Example: Element-wise function

$\mathbf{f}(\mathbf{x}) = [\sigma(x_1), \sigma(x_2), \ldots, \sigma(x_n)]^T$ (sigmoid applied element-wise)

$$J = \begin{bmatrix}
\sigma'(x_1) & 0 & \cdots & 0 \\
0 & \sigma'(x_2) & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma'(x_n)
\end{bmatrix} = \text{diag}(\sigma'(\mathbf{x}))$$

The Jacobian is diagonal for element-wise functions!

---

## The Hessian Matrix

### Definition

For $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** contains all second-order partial derivatives:

$$H = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}$$

**Note:** By Schwarz's theorem, $H$ is symmetric: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$

### What the Hessian Tells Us

- **Positive definite $H$**: Function is locally convex (minimum)
- **Negative definite $H$**: Function is locally concave (maximum)
- **Indefinite $H$**: Saddle point
- **Eigenvalues of $H$**: Curvature in different directions

### Example

$f(x, y) = x^2 + 4y^2$

First derivatives:
$$\frac{\partial f}{\partial x} = 2x, \quad \frac{\partial f}{\partial y} = 8y$$

Second derivatives:
$$H = \begin{bmatrix} 2 & 0 \\ 0 & 8 \end{bmatrix}$$

Eigenvalues: 2 and 8 (both positive → convex!)
- Curvature is 4× higher in $y$-direction
- This creates an elongated bowl (harder to optimize)

---

## Chain Rule for Multivariable Functions

### Scalar composition

If $z = f(x, y)$ where $x = g(t)$ and $y = h(t)$:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

### Vector composition (general chain rule)

If $\mathbf{z} = \mathbf{f}(\mathbf{y})$ and $\mathbf{y} = \mathbf{g}(\mathbf{x})$, then:

$$J_{\mathbf{z}/\mathbf{x}} = J_{\mathbf{f}} \cdot J_{\mathbf{g}}$$

**The chain rule becomes matrix multiplication!**

This is the foundation of backpropagation.

### Example: Two-layer network

Forward:
1. $\mathbf{h} = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$
2. $\mathbf{y} = W_2 \mathbf{h} + \mathbf{b}_2$
3. $\mathcal{L} = \ell(\mathbf{y}, \mathbf{t})$

Backward (chain rule):
$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{h}} \cdot \frac{\partial \mathbf{h}}{\partial W_1}$$

---

## Total Derivative vs Partial Derivative

**Important distinction!**

### Partial Derivative
Treats other variables as constants:
$$\frac{\partial f}{\partial x}$$ — only direct dependence on $x$

### Total Derivative
Accounts for all paths of dependence:
$$\frac{df}{dx} = \frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}\frac{dy}{dx} + \ldots$$

### Example

$f(x, y) = x^2 + y^2$ where $y = x^3$

**Partial:** $\frac{\partial f}{\partial x} = 2x$

**Total:** $\frac{df}{dx} = 2x + 2y \cdot 3x^2 = 2x + 6x^5$

In backpropagation, we compute **total derivatives** — we track all paths from loss to each parameter.

---

## Summary

| Concept | Input | Output | Result |
|---------|-------|--------|--------|
| Ordinary derivative | $\mathbb{R}$ | $\mathbb{R}$ | Scalar $f'(x)$ |
| Partial derivative | $\mathbb{R}^n$ | $\mathbb{R}$ | Scalar $\frac{\partial f}{\partial x_i}$ |
| Gradient | $\mathbb{R}^n$ | $\mathbb{R}$ | Vector $\nabla f \in \mathbb{R}^n$ |
| Jacobian | $\mathbb{R}^n$ | $\mathbb{R}^m$ | Matrix $J \in \mathbb{R}^{m \times n}$ |
| Hessian | $\mathbb{R}^n$ | $\mathbb{R}$ | Matrix $H \in \mathbb{R}^{n \times n}$ |

---

## Key Takeaways for ML

1. **Gradient** = direction of steepest ascent
2. **Jacobian** = how vector outputs depend on vector inputs
3. **Chain rule** = Jacobian multiplication
4. **Backprop** = computing Jacobian products efficiently (reverse mode)
5. **Hessian** = curvature information (used in second-order methods)
