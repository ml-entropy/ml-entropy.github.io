# Tutorial 01: Single Variable Derivatives

## What is a Derivative?

The derivative measures **rate of change** — how much a function's output changes when we slightly change its input.

### Geometric Intuition

The derivative at a point is the **slope of the tangent line** to the curve at that point.

```
     f(x)
       │    ╱
       │   ╱  ← tangent line (slope = derivative)
       │  ●──
       │ ╱
       │╱
       └──────── x
```

### Definition from First Principles

**The limit definition:**

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

**Intuition:** 
- $f(x + h) - f(x)$ = change in output (rise)
- $h$ = change in input (run)
- As $h \to 0$, the secant line becomes the tangent line

---

## Deriving Common Derivatives

### 1. Derivative of $f(x) = x^n$ (Power Rule)

**Claim:** $\frac{d}{dx}(x^n) = nx^{n-1}$

**Proof:**
$$f'(x) = \lim_{h \to 0} \frac{(x+h)^n - x^n}{h}$$

Using binomial expansion:
$$(x+h)^n = x^n + nx^{n-1}h + \frac{n(n-1)}{2}x^{n-2}h^2 + \ldots$$

Substituting:
$$f'(x) = \lim_{h \to 0} \frac{x^n + nx^{n-1}h + O(h^2) - x^n}{h}$$
$$= \lim_{h \to 0} \frac{nx^{n-1}h + O(h^2)}{h}$$
$$= \lim_{h \to 0} \left(nx^{n-1} + O(h)\right)$$
$$= \boxed{nx^{n-1}}$$

**Examples:**
- $\frac{d}{dx}(x^2) = 2x$
- $\frac{d}{dx}(x^3) = 3x^2$
- $\frac{d}{dx}(x^{-1}) = -x^{-2}$
- $\frac{d}{dx}(\sqrt{x}) = \frac{d}{dx}(x^{1/2}) = \frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$

---

### 2. Derivative of $f(x) = e^x$

**Claim:** $\frac{d}{dx}(e^x) = e^x$

This is what makes $e$ special — it's its own derivative!

**Proof using limit definition:**
$$f'(x) = \lim_{h \to 0} \frac{e^{x+h} - e^x}{h} = \lim_{h \to 0} \frac{e^x(e^h - 1)}{h} = e^x \lim_{h \to 0} \frac{e^h - 1}{h}$$

The key is: $\lim_{h \to 0} \frac{e^h - 1}{h} = 1$ (definition of $e$)

Therefore: $\boxed{\frac{d}{dx}(e^x) = e^x}$

**Why this matters in ML:**
- Softmax uses $e^x$
- Many activation functions involve exponentials
- The fact that $(e^x)' = e^x$ makes gradient computation elegant

---

### 3. Derivative of $f(x) = \ln(x)$

**Claim:** $\frac{d}{dx}(\ln x) = \frac{1}{x}$

**Proof using inverse function:**
Let $y = \ln(x)$, so $x = e^y$.

Differentiate both sides w.r.t. $x$:
$$1 = e^y \cdot \frac{dy}{dx}$$
$$\frac{dy}{dx} = \frac{1}{e^y} = \frac{1}{x}$$

Therefore: $\boxed{\frac{d}{dx}(\ln x) = \frac{1}{x}}$

**Why this matters in ML:**
- Cross-entropy loss uses $\log$
- The $1/x$ derivative creates the famous gradient scaling that prevents vanishing gradients
- Log-likelihood gradients are well-behaved because of this

---

### 4. Derivative of $f(x) = \sigma(x) = \frac{1}{1+e^{-x}}$ (Sigmoid)

**Claim:** $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**Proof:**
$$\sigma(x) = (1 + e^{-x})^{-1}$$

Using chain rule:
$$\sigma'(x) = -1 \cdot (1 + e^{-x})^{-2} \cdot (-e^{-x})$$
$$= \frac{e^{-x}}{(1 + e^{-x})^2}$$

Now, notice:
$$\sigma(x)(1 - \sigma(x)) = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \frac{e^{-x}}{(1+e^{-x})^2}$$

Therefore: $\boxed{\sigma'(x) = \sigma(x)(1 - \sigma(x))}$

**Why this matters:**
- This elegant form makes backprop through sigmoid efficient
- Maximum derivative at $\sigma(x) = 0.5$ (decision boundary)
- Derivative → 0 for extreme values (vanishing gradient!)

---

## The Chain Rule

**The most important rule for ML!**

### Statement

If $y = f(g(x))$, then:
$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)$$

### Derivation

Let $u = g(x)$, so $y = f(u)$.

For small changes:
$$\Delta y \approx \frac{df}{du} \cdot \Delta u$$
$$\Delta u \approx \frac{dg}{dx} \cdot \Delta x$$

Combining:
$$\Delta y \approx \frac{df}{du} \cdot \frac{dg}{dx} \cdot \Delta x$$

Taking the limit:
$$\frac{dy}{dx} = \frac{df}{du} \cdot \frac{dg}{dx}$$

### Examples

**Example 1:** $y = (3x + 2)^5$

Let $u = 3x + 2$, so $y = u^5$.
- $\frac{dy}{du} = 5u^4$
- $\frac{du}{dx} = 3$

$$\frac{dy}{dx} = 5u^4 \cdot 3 = 15(3x+2)^4$$

**Example 2:** $y = e^{x^2}$

Let $u = x^2$, so $y = e^u$.
- $\frac{dy}{du} = e^u$
- $\frac{du}{dx} = 2x$

$$\frac{dy}{dx} = e^{x^2} \cdot 2x = 2x e^{x^2}$$

**Example 3:** $y = \ln(\sin(x))$

Let $u = \sin(x)$, so $y = \ln(u)$.
- $\frac{dy}{du} = \frac{1}{u}$
- $\frac{du}{dx} = \cos(x)$

$$\frac{dy}{dx} = \frac{1}{\sin(x)} \cdot \cos(x) = \cot(x)$$

---

## The Product Rule

### Statement

If $y = f(x) \cdot g(x)$, then:
$$\frac{dy}{dx} = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$

### Derivation

$$\frac{dy}{dx} = \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$

Add and subtract $f(x+h)g(x)$:
$$= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)}{h}$$

$$= \lim_{h \to 0} \left[ f(x+h) \frac{g(x+h) - g(x)}{h} + g(x) \frac{f(x+h) - f(x)}{h} \right]$$

$$= f(x) \cdot g'(x) + g(x) \cdot f'(x)$$

### Example

$y = x^2 \cdot e^x$

$$\frac{dy}{dx} = 2x \cdot e^x + x^2 \cdot e^x = e^x(2x + x^2)$$

---

## The Quotient Rule

### Statement

If $y = \frac{f(x)}{g(x)}$, then:
$$\frac{dy}{dx} = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}$$

### Derivation

Write $y = f(x) \cdot [g(x)]^{-1}$ and use product rule + chain rule:

$$\frac{dy}{dx} = f'(x) \cdot [g(x)]^{-1} + f(x) \cdot (-1)[g(x)]^{-2} \cdot g'(x)$$
$$= \frac{f'(x)}{g(x)} - \frac{f(x) \cdot g'(x)}{[g(x)]^2}$$
$$= \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{[g(x)]^2}$$

---

## Summary Table

| Function | Derivative | ML Application |
|----------|------------|----------------|
| $x^n$ | $nx^{n-1}$ | Polynomial features |
| $e^x$ | $e^x$ | Softmax, activations |
| $\ln(x)$ | $1/x$ | Log-likelihood, cross-entropy |
| $\sigma(x)$ | $\sigma(1-\sigma)$ | Sigmoid activation |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | Tanh activation |
| $\text{ReLU}(x)$ | $\mathbf{1}_{x>0}$ | ReLU activation |

---

## Connection to ML

In neural networks:
1. **Forward pass**: Compute $y = f(f(f(...f(x))))$ (nested functions)
2. **Backward pass**: Apply chain rule repeatedly

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial h_n} \cdot \frac{\partial h_n}{\partial h_{n-1}} \cdots \frac{\partial h_1}{\partial \theta}$$

This is **backpropagation** — the chain rule applied systematically!
