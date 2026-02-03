# Tutorial 01: Single Variable Derivatives - Exercises

## Part A: Theory Derivations

### Exercise A1 🟢 (Easy)
**Derive the power rule from first principles**

Using the limit definition $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$:

1. Prove $\frac{d}{dx}(x^2) = 2x$
2. Prove $\frac{d}{dx}(x^3) = 3x^2$
3. Generalize to $\frac{d}{dx}(x^n) = nx^{n-1}$ using the binomial theorem

---

### Exercise A2 🟢 (Easy)
**Derive the product rule**

Using the limit definition, prove:
$$\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$$

Hint: Add and subtract $f(x+h)g(x)$ in the numerator.

---

### Exercise A3 🟡 (Medium)
**Derive the chain rule**

If $y = f(g(x))$, prove that:
$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

Use the definition:
$$\frac{dy}{dx} = \lim_{h \to 0} \frac{f(g(x+h)) - f(g(x))}{h}$$

---

### Exercise A4 🟡 (Medium)
**Derivative of sigmoid**

The sigmoid function is $\sigma(x) = \frac{1}{1 + e^{-x}}$.

1. Prove that $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
2. At what value of $x$ is $\sigma'(x)$ maximum? What is that maximum value?
3. Why does this cause vanishing gradients?

---

### Exercise A5 🔴 (Hard)
**Derivatives of inverse functions**

If $y = f(x)$ is invertible with $x = f^{-1}(y)$, prove:
$$\frac{dx}{dy} = \frac{1}{\frac{dy}{dx}}$$

Use this to:
1. Derive $\frac{d}{dx}(\ln x)$ from $\frac{d}{dx}(e^x) = e^x$
2. Derive $\frac{d}{dx}(\arcsin x)$ from $\frac{d}{dx}(\sin x) = \cos x$

---

## Part B: Coding Exercises

### Exercise B1 🟢 (Easy)
**Implement numerical differentiation**

```python
def forward_difference(f, x, h=1e-5):
    """Compute f'(x) using forward difference."""
    # YOUR CODE HERE
    pass

def central_difference(f, x, h=1e-5):
    """Compute f'(x) using central difference."""
    # YOUR CODE HERE
    pass

# Test on f(x) = x^3. Compare accuracy of both methods.
```

---

### Exercise B2 🟡 (Medium)
**Implement automatic differentiation for scalars**

```python
class DualNumber:
    """
    Dual number for forward-mode automatic differentiation.
    A dual number is a + b*ε where ε² = 0.
    
    If f(a + ε) = f(a) + f'(a)*ε, we get the derivative for free!
    """
    def __init__(self, value, derivative=0):
        self.value = value
        self.derivative = derivative
    
    def __add__(self, other):
        # YOUR CODE HERE
        pass
    
    def __mul__(self, other):
        # YOUR CODE HERE
        pass
    
    def __pow__(self, n):
        # YOUR CODE HERE
        pass

# Test: Compute derivative of f(x) = x^3 + 2x^2 + x at x = 3
```

---

### Exercise B3 🟡 (Medium)
**Visualize activation function derivatives**

```python
def plot_activation_derivatives():
    """
    For sigmoid, tanh, and ReLU:
    1. Plot the function and its derivative side by side
    2. Mark where the derivative is maximum
    3. Shade regions where |derivative| < 0.1 (vanishing gradient zone)
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B4 🔴 (Hard)
**Implement symbolic differentiation**

```python
def symbolic_derivative(expr):
    """
    Given a string expression like "x^3 + 2*x^2 + x",
    return the derivative as a string.
    
    Support: +, -, *, ^, constants, and variable x.
    
    Example: "x^3" -> "3*x^2"
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise B5 🔴 (Hard)
**Higher-order derivatives and Taylor series**

```python
def taylor_approximation(f, a, n, x):
    """
    Compute n-th order Taylor approximation of f around point a.
    
    T_n(x) = Σ_{k=0}^{n} f^(k)(a)/k! * (x-a)^k
    
    Use numerical differentiation to estimate f^(k)(a).
    
    Return the approximation evaluated at x.
    """
    # YOUR CODE HERE
    pass

# Test: Approximate sin(x) around a=0, compare to true sin(x)
```

---

## Part C: Conceptual Questions

### C1 🟢
Why is the derivative of $e^x$ equal to itself? What makes $e$ special among all possible bases?

### C2 🟢
The chain rule says $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$. Explain why this makes intuitive sense in terms of "rates of change."

### C3 🟡
Why is central difference $(f(x+h) - f(x-h))/(2h)$ more accurate than forward difference $(f(x+h) - f(x))/h$?

### C4 🟡
In neural networks, we compose many functions: $L = f_n(f_{n-1}(...f_1(x)))$. Write the derivative $\frac{dL}{dx}$ using the chain rule. Why do we compute it backward (from $f_n$ to $f_1$)?

### C5 🔴
The logarithm converts products to sums: $\log(ab) = \log(a) + \log(b)$. What does taking the derivative of both sides tell you? How is this related to the product rule?
