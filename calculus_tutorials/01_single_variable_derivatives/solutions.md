# Tutorial 01: Single Variable Derivatives - Solutions

## Part A: Theory Solutions

### Solution A1: Power Rule Derivation

**For $x^2$:**
$$f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h}$$
$$= \lim_{h \to 0} \frac{2xh + h^2}{h} = \lim_{h \to 0} (2x + h) = \boxed{2x}$$

**For $x^3$:**
$$f'(x) = \lim_{h \to 0} \frac{(x+h)^3 - x^3}{h}$$

Expand: $(x+h)^3 = x^3 + 3x^2h + 3xh^2 + h^3$

$$= \lim_{h \to 0} \frac{3x^2h + 3xh^2 + h^3}{h} = \lim_{h \to 0} (3x^2 + 3xh + h^2) = \boxed{3x^2}$$

**General case using binomial theorem:**
$$(x+h)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} h^k = x^n + nx^{n-1}h + O(h^2)$$

$$f'(x) = \lim_{h \to 0} \frac{x^n + nx^{n-1}h + O(h^2) - x^n}{h} = \lim_{h \to 0} (nx^{n-1} + O(h)) = \boxed{nx^{n-1}}$$

---

### Solution A2: Product Rule

$$\frac{d}{dx}[fg] = \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$

Add and subtract $f(x+h)g(x)$:
$$= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)}{h}$$

$$= \lim_{h \to 0} \left[f(x+h)\frac{g(x+h) - g(x)}{h} + g(x)\frac{f(x+h) - f(x)}{h}\right]$$

Taking limits:
$$= f(x)g'(x) + g(x)f'(x) = \boxed{f'g + fg'}$$

---

### Solution A3: Chain Rule

Let $u = g(x)$. Then:
$$\frac{dy}{dx} = \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{h}$$

Multiply and divide by $g(x+h) - g(x)$ (when non-zero):
$$= \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{g(x+h) - g(x)} \cdot \frac{g(x+h) - g(x)}{h}$$

As $h \to 0$: $g(x+h) - g(x) \to 0$ (if $g$ is continuous), so:
$$= f'(g(x)) \cdot g'(x)$$

Therefore: $\boxed{\frac{dy}{dx} = f'(g(x)) \cdot g'(x)}$

---

### Solution A4: Sigmoid Derivative

**Part 1:**
$$\sigma(x) = (1 + e^{-x})^{-1}$$

Using chain rule:
$$\sigma'(x) = -1 \cdot (1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2}$$

Now observe:
$$\sigma(x)(1-\sigma(x)) = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \frac{e^{-x}}{(1+e^{-x})^2}$$

Therefore: $\boxed{\sigma'(x) = \sigma(x)(1-\sigma(x))}$

**Part 2:**
$\sigma'(x) = \sigma(1-\sigma)$ is maximized when $\sigma = 0.5$, i.e., when $x = 0$.
Maximum value: $0.5 \times 0.5 = \boxed{0.25}$

**Part 3:**
When $|x|$ is large, $\sigma \approx 0$ or $\sigma \approx 1$, so $\sigma' \approx 0$. Gradients become very small for extreme inputs, preventing learning (vanishing gradients).

---

### Solution A5: Inverse Function Derivatives

If $y = f(x)$, then $x = f^{-1}(y)$.

Differentiate $y = f(f^{-1}(y))$ with respect to $y$:
$$1 = f'(f^{-1}(y)) \cdot \frac{d(f^{-1})}{dy}$$

Therefore: $\frac{dx}{dy} = \frac{1}{f'(x)}$ or $\boxed{\frac{d(f^{-1})}{dy} = \frac{1}{f'(f^{-1}(y))}}$

**For $\ln x$:**
$y = \ln x \Leftrightarrow x = e^y$

$\frac{dx}{dy} = e^y = x$, so $\frac{dy}{dx} = \frac{1}{x}$. Therefore: $\boxed{\frac{d}{dx}(\ln x) = \frac{1}{x}}$

**For $\arcsin x$:**
$y = \arcsin x \Leftrightarrow x = \sin y$

$\frac{dx}{dy} = \cos y = \sqrt{1 - \sin^2 y} = \sqrt{1 - x^2}$

Therefore: $\boxed{\frac{d}{dx}(\arcsin x) = \frac{1}{\sqrt{1-x^2}}}$

---

## Part B: Coding Solutions

### Solution B1: Numerical Differentiation

```python
import numpy as np

def forward_difference(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Test
f = lambda x: x**3
true_derivative = lambda x: 3*x**2

x = 2.0
true_val = true_derivative(x)
forward = forward_difference(f, x)
central = central_difference(f, x)

print(f"True derivative at x=2: {true_val}")
print(f"Forward difference: {forward}, error: {abs(forward - true_val):.2e}")
print(f"Central difference: {central}, error: {abs(central - true_val):.2e}")

# Central difference is more accurate (O(h²) vs O(h) error)
```

---

### Solution B2: Dual Numbers

```python
class DualNumber:
    def __init__(self, value, derivative=0):
        self.value = value
        self.derivative = derivative
    
    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value + other.value, 
                            self.derivative + other.derivative)
        return DualNumber(self.value + other, self.derivative)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # (a + b*ε)(c + d*ε) = ac + (ad + bc)*ε
            return DualNumber(self.value * other.value,
                            self.value * other.derivative + self.derivative * other.value)
        return DualNumber(self.value * other, self.derivative * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, n):
        # d/dx(x^n) = n*x^(n-1) * dx
        return DualNumber(self.value ** n, 
                         n * self.value ** (n-1) * self.derivative)
    
    def __repr__(self):
        return f"DualNumber({self.value}, {self.derivative})"

# Test: f(x) = x³ + 2x² + x at x = 3
# f'(x) = 3x² + 4x + 1 = 27 + 12 + 1 = 40
x = DualNumber(3, 1)  # x with derivative 1 (seed)
result = x**3 + 2*x**2 + x
print(f"f(3) = {result.value}")
print(f"f'(3) = {result.derivative}")  # Should be 40
```

---

### Solution B3: Activation Derivatives Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_activation_derivatives():
    x = np.linspace(-5, 5, 200)
    
    activations = {
        'Sigmoid': (1/(1+np.exp(-x)), (1/(1+np.exp(-x)))*(1-1/(1+np.exp(-x)))),
        'Tanh': (np.tanh(x), 1 - np.tanh(x)**2),
        'ReLU': (np.maximum(0, x), (x > 0).astype(float))
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    for i, (name, (f, df)) in enumerate(activations.items()):
        # Function
        axes[i, 0].plot(x, f, 'b-', linewidth=2)
        axes[i, 0].set_title(f'{name}: f(x)')
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(0, color='black', linewidth=0.5)
        
        # Derivative
        axes[i, 1].plot(x, df, 'r-', linewidth=2)
        axes[i, 1].set_title(f"{name}: f'(x)")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(0, color='black', linewidth=0.5)
        
        # Mark max derivative
        max_idx = np.argmax(df)
        axes[i, 1].scatter([x[max_idx]], [df[max_idx]], color='green', s=100, zorder=5)
        
        # Shade vanishing gradient zone
        vanish = np.abs(df) < 0.1
        axes[i, 1].fill_between(x, 0, df.max(), where=vanish, alpha=0.3, color='red')
    
    plt.tight_layout()
    plt.show()

plot_activation_derivatives()
```

---

### Solution B4: Symbolic Differentiation

```python
import re

def symbolic_derivative(expr):
    """Simple symbolic differentiator for polynomials."""
    # Remove spaces
    expr = expr.replace(' ', '')
    
    # Split into terms (keep + and -)
    terms = re.split(r'(?=[+-])', expr)
    terms = [t for t in terms if t]
    
    derivatives = []
    
    for term in terms:
        term = term.strip()
        
        # Handle x^n
        match = re.match(r'([+-]?\d*)?\*?x\^(\d+)', term)
        if match:
            coef = int(match.group(1)) if match.group(1) and match.group(1) not in ['+', '-'] else (1 if not match.group(1) or match.group(1) == '+' else -1)
            power = int(match.group(2))
            new_coef = coef * power
            new_power = power - 1
            if new_power == 0:
                derivatives.append(str(new_coef))
            elif new_power == 1:
                derivatives.append(f"{new_coef}*x")
            else:
                derivatives.append(f"{new_coef}*x^{new_power}")
            continue
        
        # Handle x (x^1)
        match = re.match(r'([+-]?\d*)?\*?x$', term)
        if match:
            coef = int(match.group(1)) if match.group(1) and match.group(1) not in ['+', '-'] else (1 if not match.group(1) or match.group(1) == '+' else -1)
            derivatives.append(str(coef))
            continue
        
        # Constants differentiate to 0
    
    result = '+'.join(derivatives).replace('+-', '-')
    return result if result else '0'

# Test
print(symbolic_derivative("x^3"))  # 3*x^2
print(symbolic_derivative("x^3+2*x^2+x"))  # 3*x^2+4*x+1
print(symbolic_derivative("5*x^4-3*x^2+x"))  # 20*x^3-6*x+1
```

---

### Solution B5: Taylor Approximation

```python
import numpy as np
from math import factorial

def nth_derivative(f, x, n, h=1e-3):
    """Estimate n-th derivative numerically using finite differences."""
    if n == 0:
        return f(x)
    # Use central difference recursively
    return (nth_derivative(f, x+h, n-1, h) - nth_derivative(f, x-h, n-1, h)) / (2*h)

def taylor_approximation(f, a, n, x):
    """Compute n-th order Taylor approximation at point x around a."""
    result = 0
    for k in range(n + 1):
        deriv_k = nth_derivative(f, a, k)
        result += deriv_k / factorial(k) * (x - a)**k
    return result

# Test: Approximate sin(x) around a=0
x_vals = np.linspace(-np.pi, np.pi, 100)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, np.sin(x_vals), 'b-', linewidth=2, label='sin(x)')

for n in [1, 3, 5, 7]:
    approx = [taylor_approximation(np.sin, 0, n, x) for x in x_vals]
    plt.plot(x_vals, approx, '--', label=f'Taylor order {n}')

plt.ylim(-2, 2)
plt.legend()
plt.grid(True)
plt.title('Taylor Approximations of sin(x)')
plt.show()
```

---

## Part C: Conceptual Answers

### C1: Why $e^x$ is special
$e$ is defined as the unique base where $\frac{d}{dx}b^x = b^x$. For any other base $b$:
$$\frac{d}{dx}b^x = b^x \ln(b)$$

Only when $b = e$ do we get $\ln(e) = 1$, so the derivative equals the function.

### C2: Chain rule intuition
If $y$ changes 3× as fast as $u$, and $u$ changes 2× as fast as $x$, then $y$ changes $3 \times 2 = 6$× as fast as $x$. The chain rule multiplies these "rates of change" together.

### C3: Central vs Forward difference
Forward difference error: $O(h)$ (linear in $h$)
Central difference error: $O(h^2)$ (quadratic in $h$)

Central difference uses symmetric points, so odd-order errors cancel out, leaving only even-order errors.

### C4: Backpropagation
$$\frac{dL}{dx} = f'_n(z_n) \cdot f'_{n-1}(z_{n-1}) \cdot \ldots \cdot f'_1(x)$$

We compute backward because:
1. We only need gradient w.r.t. parameters (not intermediate values)
2. Computing forward would require one pass per parameter (expensive!)
3. Backward: one pass gives all gradients (efficient)

### C5: Log and product rule
Taking $\frac{d}{dx}$ of $\log(fg) = \log f + \log g$:
$$\frac{1}{fg} \cdot \frac{d(fg)}{dx} = \frac{f'}{f} + \frac{g'}{g}$$

Multiply by $fg$:
$$\frac{d(fg)}{dx} = f'g + fg'$$

This gives an alternative derivation of the product rule!
