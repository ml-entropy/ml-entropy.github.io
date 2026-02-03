# Tutorial 05: Logarithm in ML — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Log Properties

a) **Product to sum**: Let $\log(a) = x$, $\log(b) = y$. Then $a = e^x$, $b = e^y$.
$$ab = e^x \cdot e^y = e^{x+y}$$
$$\log(ab) = x + y = \log(a) + \log(b)$$ ∎

b) **Power rule**: $a^n = (e^{\log a})^n = e^{n\log a}$
$$\log(a^n) = n\log(a)$$ ∎

c) **Change of base**: Let $\log_a(b) = x$, so $a^x = b$.
Taking $\log_c$ of both sides:
$$\log_c(a^x) = \log_c(b)$$
$$x\log_c(a) = \log_c(b)$$
$$x = \frac{\log_c(b)}{\log_c(a)}$$ ∎

---

### Solution A2 — Why Log-Likelihood?

1. **Numerical stability**: Products of small numbers underflow; sums don't
2. **Computational simplicity**: Products become sums, easier to differentiate
3. **Same optima**: $\arg\max_\theta L(\theta) = \arg\max_\theta \log L(\theta)$ since log is monotonic
4. **Connection to information**: $-\log p(x)$ is the "surprise" or information content

---

### Solution A3 — Gradient Comparison

For $L(\theta) = \prod_{i=1}^n p_i$ where $p_i = p(x_i|\theta)$:

a) **Likelihood gradient** (product rule):
$$\frac{\partial L}{\partial \theta} = \sum_{i=1}^n \left(\frac{\partial p_i}{\partial \theta} \prod_{j \neq i} p_j\right) = L \sum_{i=1}^n \frac{1}{p_i}\frac{\partial p_i}{\partial \theta}$$

b) **Log-likelihood gradient**:
$$\frac{\partial \log L}{\partial \theta} = \frac{\partial}{\partial \theta} \sum_i \log p_i = \sum_i \frac{1}{p_i}\frac{\partial p_i}{\partial \theta}$$

c) **Same zeros**: Setting derivatives to zero:
- Likelihood: $L \sum_i \frac{1}{p_i}\frac{\partial p_i}{\partial \theta} = 0$
- Log-likelihood: $\sum_i \frac{1}{p_i}\frac{\partial p_i}{\partial \theta} = 0$

Since $L > 0$, both equations have the same solutions. ∎

---

### Solution A4 — Shannon's Uniqueness

We want $I: (0,1] \to \mathbb{R}_{\geq 0}$ with:
1. $I(p) \geq 0$
2. $I(1) = 0$
3. $I(pq) = I(p) + I(q)$ (additivity)

**Proof sketch**:

From (3) with $p = q$: $I(p^2) = 2I(p)$

By induction: $I(p^n) = nI(p)$

For rational $m/n$: $I(p^{m/n}) = \frac{m}{n}I(p)$

By continuity (assumed): $I(p^r) = rI(p)$ for all real $r$.

Let $I(e^{-1}) = k$ for some constant $k$. Then:
$$I(p) = I(e^{\ln p}) = I((e^{-1})^{-\ln p}) = -\ln(p) \cdot I(e^{-1}) = -k\ln(p)$$

From (1): $k > 0$ (since $\ln p < 0$ for $p < 1$).

$$\boxed{I(p) = -k\log(p)}$$

---

### Solution A5 — Log-Sum-Exp Stability

a) **Problem**: For $x_i$ large positive, $e^{x_i}$ overflows. For $x_i$ large negative, underflows to 0.

b) **Stable formula**: Let $m = \max_i x_i$
$$\log\sum_i e^{x_i} = \log\sum_i e^{x_i - m + m} = \log\left(e^m \sum_i e^{x_i - m}\right) = m + \log\sum_i e^{x_i - m}$$

Now $x_i - m \leq 0$, so $e^{x_i - m} \leq 1$ (no overflow).
At least one term equals $e^0 = 1$ (no underflow to 0).

$$\boxed{\text{logsumexp}(x) = \max(x) + \log\sum_i e^{x_i - \max(x)}}$$

---

### Solution A6 — Is Log Fundamental?

**Yes, log is mathematically fundamental**, not just a computational convenience:

1. **Gradient scaling**: Log-likelihood gradient is $\sum_i \frac{\partial \log p_i}{\partial \theta}$, independent of other samples. Without log, gradient includes $\prod_{j \neq i} p_j$, which vanishes as $n \to \infty$.

2. **Information theory**: Shannon proved log is the ONLY function satisfying additivity + non-negativity. Entropy must use log.

3. **Natural parameterization**: Exponential family distributions have log-likelihood linear in sufficient statistics — this is fundamental to efficient inference.

Even with infinite precision, without log:
- Gradients would be astronomically small (exponentially many multiplications)
- No natural connection between likelihood and information
- Loss of the additive structure that makes SGD work

---

## Part B: Coding Solutions

### Solution B1 — Numerical Demonstration

```python
import numpy as np

# 1000 probabilities, each ~0.1
n = 1000
probs = np.full(n, 0.1)

# Direct product
product = np.prod(probs)
print(f"Direct product: {product}")  # 0.0 (underflow!)

# Log-sum approach
log_sum = np.sum(np.log(probs))
print(f"Log sum: {log_sum:.2f}")  # -2302.59
print(f"Recovered product: {np.exp(log_sum)}")  # Still 0 (too small for float64)
print(f"Log10 of product: {log_sum / np.log(10):.2f}")  # -1000 (10^-1000)

# The answer is 0.1^1000 = 10^(-1000), which is unrepresentable
# But log tells us the magnitude!
```

### Solution B2 — Log-Sum-Exp

```python
import numpy as np

def logsumexp_naive(x):
    """Naive implementation - can overflow/underflow"""
    return np.log(np.sum(np.exp(x)))

def logsumexp_stable(x):
    """Numerically stable implementation"""
    x = np.asarray(x)
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))

# Test on normal values
x_normal = [1, 2, 3]
print(f"Normal input: naive={logsumexp_naive(x_normal):.4f}, stable={logsumexp_stable(x_normal):.4f}")

# Test on large values (overflow case)
x_large = [1000, 1001, 1002]
try:
    naive = logsumexp_naive(x_large)
    print(f"Large input naive: {naive}")
except:
    print("Naive failed on large input!")
print(f"Large input stable: {logsumexp_stable(x_large):.4f}")  # Works!

# Test on small values (underflow case)
x_small = [-1000, -1001, -1002]
print(f"Small input naive: {logsumexp_naive(x_small)}")  # -inf (wrong!)
print(f"Small input stable: {logsumexp_stable(x_small):.4f}")  # Correct!

# Verify with scipy
from scipy.special import logsumexp as scipy_lse
print(f"Scipy reference: {scipy_lse(x_large):.4f}")
```

### Solution B3 — MLE Without Log (Symbolic)

```python
import sympy as sp

# Define symbols
mu, sigma = sp.symbols('mu sigma', real=True, positive=True)
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)

# Gaussian PDF
def gaussian_pdf(x, mu, sigma):
    return 1/(sp.sqrt(2*sp.pi)*sigma) * sp.exp(-(x-mu)**2 / (2*sigma**2))

# Likelihood (product of 3 samples)
L = gaussian_pdf(x1, mu, sigma) * gaussian_pdf(x2, mu, sigma) * gaussian_pdf(x3, mu, sigma)
L_simplified = sp.simplify(L)

# Log-likelihood (sum)
log_L = sp.log(gaussian_pdf(x1, mu, sigma)) + sp.log(gaussian_pdf(x2, mu, sigma)) + sp.log(gaussian_pdf(x3, mu, sigma))
log_L_simplified = sp.simplify(log_L)

print("=== LIKELIHOOD GRADIENT ===")
dL_dmu = sp.diff(L, mu)
print(f"∂L/∂μ has {sp.count_ops(dL_dmu)} operations")

print("\n=== LOG-LIKELIHOOD GRADIENT ===")
dlogL_dmu = sp.diff(log_L_simplified, mu)
print(f"∂log(L)/∂μ = {sp.simplify(dlogL_dmu)}")
print(f"∂log(L)/∂μ has {sp.count_ops(sp.simplify(dlogL_dmu))} operations")

# The log-likelihood gradient is MUCH simpler!
# ∂log(L)/∂μ = (x1 + x2 + x3 - 3μ) / σ²
```

---

## Part C: Conceptual Solutions

### C1

Cross-entropy loss $-\sum_i y_i \log(\hat{y}_i)$ uses log because:

1. **Gradient magnitude**: $\frac{\partial}{\partial \hat{y}} (-\log \hat{y}) = -\frac{1}{\hat{y}}$
   - When prediction is wrong ($\hat{y}$ small), gradient is large → fast learning
   - Without log: gradient could be tiny when $\hat{y}$ is small

2. **Connection to probability**: Log-likelihood is the natural loss for probabilistic models

3. **Prevents saturation**: For sigmoid/softmax outputs near 0 or 1, log loss still provides useful gradients

### C2

The log function is the **bridge** between:

1. **Information theory**: $I(x) = -\log p(x)$ is information content. Entropy is expected information.

2. **Probability**: Log-likelihood converts products to sums, enabling MLE via gradient descent.

3. **Neural networks**: 
   - Cross-entropy loss = KL divergence + constant
   - Softmax + log = log-softmax (numerically stable)
   - SGD works because log-likelihood gradients are additive over samples

**The unifying principle**: Independence in probability ($p(A,B) = p(A)p(B)$) becomes additivity in log space ($\log p(A,B) = \log p(A) + \log p(B)$), which is what gradient-based optimization needs.
