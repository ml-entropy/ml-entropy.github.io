# Tutorial 03: Directional Derivatives

## Motivation

The partial derivative tells us the rate of change along coordinate axes. But what if we want to know the rate of change in **any arbitrary direction**?

---

## Definition

For a function $f: \mathbb{R}^n \to \mathbb{R}$ at point $\mathbf{x}$, the **directional derivative** in direction $\mathbf{v}$ is:

$$D_{\mathbf{v}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{v}) - f(\mathbf{x})}{h}$$

where $\mathbf{v}$ is a **unit vector** ($\|\mathbf{v}\| = 1$).

### Intuition

- Start at point $\mathbf{x}$
- Take a tiny step in direction $\mathbf{v}$
- Measure how much $f$ changes

---

## The Fundamental Formula

### Theorem

$$D_{\mathbf{v}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{v}$$

The directional derivative is the **dot product** of the gradient with the direction vector!

### Derivation

Let $\mathbf{x}(t) = \mathbf{x}_0 + t\mathbf{v}$ be a parametric line through $\mathbf{x}_0$ in direction $\mathbf{v}$.

Define $g(t) = f(\mathbf{x}(t))$.

By the chain rule:
$$g'(t) = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot \frac{dx_i}{dt}$$

Since $x_i(t) = x_{0,i} + t v_i$, we have $\frac{dx_i}{dt} = v_i$.

Therefore:
$$g'(t) = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} \cdot v_i = \nabla f \cdot \mathbf{v}$$

At $t = 0$ (at point $\mathbf{x}_0$):
$$\boxed{D_{\mathbf{v}} f(\mathbf{x}_0) = \nabla f(\mathbf{x}_0) \cdot \mathbf{v}}$$

---

## Special Cases

### 1. Along coordinate axes

Let $\mathbf{e}_i$ be the $i$-th standard basis vector (all zeros except 1 in position $i$).

$$D_{\mathbf{e}_i} f = \nabla f \cdot \mathbf{e}_i = \frac{\partial f}{\partial x_i}$$

**Partial derivatives are special cases of directional derivatives!**

### 2. In direction of gradient

Let $\mathbf{v} = \frac{\nabla f}{\|\nabla f\|}$ (unit vector in gradient direction).

$$D_{\mathbf{v}} f = \nabla f \cdot \frac{\nabla f}{\|\nabla f\|} = \frac{\|\nabla f\|^2}{\|\nabla f\|} = \|\nabla f\|$$

**The directional derivative is maximized in the gradient direction!**

### 3. Opposite to gradient

Let $\mathbf{v} = -\frac{\nabla f}{\|\nabla f\|}$.

$$D_{\mathbf{v}} f = -\|\nabla f\|$$

**This is the direction of steepest descent — exactly what gradient descent uses!**

---

## Why Gradient is Steepest Ascent (Proof)

### Theorem

Among all unit vectors $\mathbf{v}$, the directional derivative $D_{\mathbf{v}} f$ is maximized when $\mathbf{v} = \frac{\nabla f}{\|\nabla f\|}$.

### Proof

$$D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v} = \|\nabla f\| \|\mathbf{v}\| \cos\theta = \|\nabla f\| \cos\theta$$

where $\theta$ is the angle between $\nabla f$ and $\mathbf{v}$.

Since $-1 \leq \cos\theta \leq 1$:
- **Maximum** when $\theta = 0$ (same direction): $D_{\mathbf{v}} f = \|\nabla f\|$
- **Minimum** when $\theta = \pi$ (opposite direction): $D_{\mathbf{v}} f = -\|\nabla f\|$
- **Zero** when $\theta = \pi/2$ (perpendicular): $D_{\mathbf{v}} f = 0$

---

## Level Curves and Gradient

### Key Insight

**The gradient is perpendicular to level curves (contour lines).**

### Proof

A level curve is defined by $f(\mathbf{x}) = c$ for some constant $c$.

If $\mathbf{x}(t)$ is a curve along this level set, then $f(\mathbf{x}(t)) = c$ for all $t$.

Differentiating:
$$\frac{d}{dt} f(\mathbf{x}(t)) = \nabla f \cdot \mathbf{x}'(t) = 0$$

This means $\nabla f \perp \mathbf{x}'(t)$, where $\mathbf{x}'(t)$ is tangent to the level curve.

### Visual

```
Level curves of f(x,y) = x² + y²

        y
        │    ╱ ∇f (points outward)
        │   ╱
    ────●────── level curve (circle)
        │
        │
        └──────── x
```

The gradient points perpendicular to the circles, toward the center (the minimum).

---

## Example: 2D Function

$f(x, y) = x^2 + 4y^2$

### Gradient
$$\nabla f = \begin{bmatrix} 2x \\ 8y \end{bmatrix}$$

### At point $(1, 1)$
$$\nabla f(1, 1) = \begin{bmatrix} 2 \\ 8 \end{bmatrix}$$

### Directional derivative in direction $\mathbf{v} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$

$$D_{\mathbf{v}} f = \begin{bmatrix} 2 \\ 8 \end{bmatrix} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{2}}(2 + 8) = \frac{10}{\sqrt{2}} \approx 7.07$$

### Maximum directional derivative

Direction: $\frac{\nabla f}{\|\nabla f\|} = \frac{1}{\sqrt{68}}\begin{bmatrix} 2 \\ 8 \end{bmatrix}$

Magnitude: $\|\nabla f\| = \sqrt{4 + 64} = \sqrt{68} \approx 8.25$

---

## Applications in ML

### 1. Gradient Descent

Move in direction of steepest descent:
$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}$$

This is choosing $\mathbf{v} = -\frac{\nabla \mathcal{L}}{\|\nabla \mathcal{L}\|}$ (but with learning rate $\eta$ controlling step size).

### 2. Natural Gradient

Sometimes the steepest direction in parameter space isn't the best. Natural gradient considers the geometry of the probability distribution space.

### 3. Constrained Optimization

When optimizing on a constraint surface, we project the gradient onto the tangent plane. This uses directional derivatives along the constraint.

---

## Second-Order Directional Derivative

The second directional derivative is:
$$D_{\mathbf{v}}^2 f = D_{\mathbf{v}}(D_{\mathbf{v}} f) = \mathbf{v}^T H \mathbf{v}$$

where $H$ is the Hessian.

### Interpretation
- Positive: Function curves upward in direction $\mathbf{v}$ (convex locally)
- Negative: Function curves downward (concave locally)
- Zero: Linear in that direction (at that point)

---

## Summary

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| Directional derivative | $D_{\mathbf{v}} f = \nabla f \cdot \mathbf{v}$ | Rate of change in direction $\mathbf{v}$ |
| Maximum rate | $\|\nabla f\|$ | In direction of $\nabla f$ |
| Minimum rate | $-\|\nabla f\|$ | In direction of $-\nabla f$ |
| Zero rate | $0$ | Perpendicular to $\nabla f$ (along level curve) |

---

## Key Takeaways

1. **Directional derivative** = dot product with gradient
2. **Gradient** = direction of steepest ascent
3. **Gradient** ⊥ level curves
4. **Gradient descent** moves opposite to gradient (steepest descent)
5. The magnitude $\|\nabla f\|$ tells us how fast the function changes in the steepest direction
