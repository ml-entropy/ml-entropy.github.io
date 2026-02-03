# Tutorial 01: Entropy Fundamentals — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Information of Events

Formula: $I(x) = -\log_2 P(x)$ bits

a) Fair coin heads: $I = -\log_2(0.5) = 1$ bit

b) Rolling a 6: $I = -\log_2(1/6) = \log_2(6) \approx 2.585$ bits

c) Certain event: $I = -\log_2(1) = 0$ bits (no surprise!)

d) Drawing an ace: $I = -\log_2(4/52) = -\log_2(1/13) = \log_2(13) \approx 3.70$ bits

---

### Solution A2 — Entropy Calculation

Formula: $H(X) = -\sum_i p_i \log_2 p_i$

a) **Fair coin**:
$$H = -0.5\log_2(0.5) - 0.5\log_2(0.5) = -2 \times 0.5 \times (-1) = 1 \text{ bit}$$

b) **Biased coin** (p=0.9):
$$H = -0.9\log_2(0.9) - 0.1\log_2(0.1)$$
$$= -0.9 \times (-0.152) - 0.1 \times (-3.322)$$
$$= 0.137 + 0.332 = 0.469 \text{ bits}$$

c) **Fair 8-sided die**:
$$H = -8 \times \frac{1}{8}\log_2\frac{1}{8} = -\log_2\frac{1}{8} = \log_2(8) = 3 \text{ bits}$$

d) **Uniform over 4**:
$$H = -4 \times 0.25 \times \log_2(0.25) = -\log_2(0.25) = 2 \text{ bits}$$

---

### Solution A3 — Entropy Maximization

a) **Maximum entropy distribution**: Uniform distribution $P = [0.25, 0.25, 0.25, 0.25]$

**Proof using Lagrange multipliers**:

Maximize $H = -\sum_i p_i \ln p_i$ subject to $\sum_i p_i = 1$

Lagrangian: $\mathcal{L} = -\sum_i p_i \ln p_i - \lambda(\sum_i p_i - 1)$

$$\frac{\partial \mathcal{L}}{\partial p_i} = -\ln p_i - 1 - \lambda = 0$$

$$\ln p_i = -(1 + \lambda) \quad \Rightarrow \quad p_i = e^{-(1+\lambda)}$$

All $p_i$ are equal! With constraint $\sum p_i = 1$: $p_i = 1/n$. ∎

b) **Maximum entropy**: $H_{max} = \log_2(4) = 2$ bits

c) **General formula**: For $n$ outcomes, $H_{max} = \log_2(n)$ bits

---

### Solution A4 — Cross-Entropy

Given: $P = [0.7, 0.2, 0.1]$, $Q = [0.5, 0.3, 0.2]$

a) **Entropy** $H(P)$:
$$H(P) = -0.7\log_2(0.7) - 0.2\log_2(0.2) - 0.1\log_2(0.1)$$
$$= -0.7(-0.515) - 0.2(-2.322) - 0.1(-3.322)$$
$$= 0.360 + 0.464 + 0.332 = 1.157 \text{ bits}$$

b) **Cross-entropy** $H(P, Q)$:
$$H(P, Q) = -\sum_i P(i) \log_2 Q(i)$$
$$= -0.7\log_2(0.5) - 0.2\log_2(0.3) - 0.1\log_2(0.2)$$
$$= 0.7(1) + 0.2(1.737) + 0.1(2.322)$$
$$= 0.7 + 0.347 + 0.232 = 1.280 \text{ bits}$$

c) **KL divergence**:
$$D_{KL}(P||Q) = H(P, Q) - H(P) = 1.280 - 1.157 = 0.123 \text{ bits}$$

Or directly:
$$D_{KL} = \sum_i P(i) \log_2 \frac{P(i)}{Q(i)} = 0.7\log_2\frac{0.7}{0.5} + 0.2\log_2\frac{0.2}{0.3} + 0.1\log_2\frac{0.1}{0.2}$$
$$= 0.7(0.485) + 0.2(-0.585) + 0.1(-1) = 0.340 - 0.117 - 0.100 = 0.123 \text{ bits}$$

d) **Verification**: $H(P, Q) = H(P) + D_{KL}(P||Q)$
$$1.280 = 1.157 + 0.123 \checkmark$$

---

### Solution A5 — Joint and Conditional Entropy

a) **Marginals**:
- $P(X=0) = 0.4 + 0.1 = 0.5$, $P(X=1) = 0.5$
- $P(Y=0) = 0.4 + 0.1 = 0.5$, $P(Y=1) = 0.5$

$$H(X) = H(Y) = -0.5\log_2(0.5) - 0.5\log_2(0.5) = 1 \text{ bit}$$

b) **Joint entropy**:
$$H(X, Y) = -0.4\log_2(0.4) - 0.1\log_2(0.1) - 0.1\log_2(0.1) - 0.4\log_2(0.4)$$
$$= -2(0.4)(-1.322) - 2(0.1)(-3.322)$$
$$= 1.058 + 0.664 = 1.722 \text{ bits}$$

c) **Conditional entropy**:
$$H(X|Y) = H(X, Y) - H(Y) = 1.722 - 1 = 0.722 \text{ bits}$$

Or compute directly using $H(X|Y) = \sum_y P(y) H(X|Y=y)$:
- $H(X|Y=0) = -\frac{0.4}{0.5}\log_2\frac{0.4}{0.5} - \frac{0.1}{0.5}\log_2\frac{0.1}{0.5} = -0.8\log_2(0.8) - 0.2\log_2(0.2) = 0.722$
- $H(X|Y=1) = 0.722$ (by symmetry)
- $H(X|Y) = 0.5(0.722) + 0.5(0.722) = 0.722$ ✓

d) **Verification**: $H(X, Y) = H(Y) + H(X|Y) = 1 + 0.722 = 1.722$ ✓

---

### Solution A6 — Derive Information Formula

**Given axioms**:
1. $I(p)$ continuous
2. $I(p) \geq 0$
3. $I(1) = 0$
4. $I(p_1 \cdot p_2) = I(p_1) + I(p_2)$ (additivity)

**Derivation**:

Let $f(x) = I(e^{-x})$ for $x \geq 0$ (substituting $p = e^{-x}$).

From axiom 4:
$$I(e^{-x_1} \cdot e^{-x_2}) = I(e^{-x_1}) + I(e^{-x_2})$$
$$I(e^{-(x_1+x_2)}) = I(e^{-x_1}) + I(e^{-x_2})$$
$$f(x_1 + x_2) = f(x_1) + f(x_2)$$

This is Cauchy's functional equation! The only continuous solutions are:
$$f(x) = cx$$

for some constant $c$.

Therefore:
$$I(e^{-x}) = cx$$
$$I(p) = c \cdot (-\ln p) = -c \ln p$$

From axiom 2, $I(p) \geq 0$ for $p \in (0, 1]$, so $c > 0$.

**Choosing $c = 1/\ln(2)$ gives**: $I(p) = -\log_2(p)$ ∎

---

### Solution A7 — Maximum Entropy Principle

**Goal**: Find distribution maximizing $h(p) = -\int_0^\infty p(x) \ln p(x) dx$ subject to:
- $\int_0^\infty p(x) dx = 1$
- $\int_0^\infty x \, p(x) dx = \mu$

**Lagrangian**:
$$\mathcal{L} = -\int p \ln p \, dx - \lambda_0(\int p \, dx - 1) - \lambda_1(\int x p \, dx - \mu)$$

**Functional derivative**:
$$\frac{\delta \mathcal{L}}{\delta p} = -\ln p - 1 - \lambda_0 - \lambda_1 x = 0$$

$$\ln p = -(1 + \lambda_0) - \lambda_1 x$$
$$p(x) = e^{-(1+\lambda_0)} e^{-\lambda_1 x} = A e^{-\lambda_1 x}$$

This is the **exponential distribution**!

From $\int_0^\infty A e^{-\lambda_1 x} dx = 1$: $A = \lambda_1$

From $\int_0^\infty x \lambda_1 e^{-\lambda_1 x} dx = \mu$: $\lambda_1 = 1/\mu$

**Result**: $p(x) = \frac{1}{\mu} e^{-x/\mu}$ (Exponential with mean $\mu$) ∎

---

### Solution A8 — Entropy of Gaussian

For $X \sim N(\mu, \sigma^2)$, $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

$$h(X) = -\int_{-\infty}^{\infty} p(x) \ln p(x) dx$$

$$\ln p(x) = -\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

$$h(X) = -\int p(x) \left[-\frac{1}{2}\ln(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}\right] dx$$

$$= \frac{1}{2}\ln(2\pi\sigma^2) \underbrace{\int p(x) dx}_{=1} + \frac{1}{2\sigma^2} \underbrace{\int (x-\mu)^2 p(x) dx}_{=\sigma^2}$$

$$= \frac{1}{2}\ln(2\pi\sigma^2) + \frac{\sigma^2}{2\sigma^2}$$

$$= \frac{1}{2}\ln(2\pi\sigma^2) + \frac{1}{2}$$

$$\boxed{h(X) = \frac{1}{2}\ln(2\pi e \sigma^2)}$$

Note: Only depends on $\sigma$, not $\mu$ (shifting doesn't change uncertainty). ∎

---

## Part B: Coding Solutions

### Solution B1 — Entropy Calculator

```python
import numpy as np

def entropy(p, base=2):
    """
    Compute entropy of distribution p.
    
    Args:
        p: probability distribution (array-like, should sum to 1)
        base: logarithm base (2 for bits, e for nats)
    
    Returns:
        H(X): entropy value
    """
    p = np.asarray(p, dtype=float)
    
    # Validate
    assert np.all(p >= 0), "Probabilities must be non-negative"
    assert np.isclose(p.sum(), 1), f"Probabilities must sum to 1, got {p.sum()}"
    
    # Handle p=0 case: 0 * log(0) = 0
    p_nonzero = p[p > 0]
    
    if base == 2:
        return -np.sum(p_nonzero * np.log2(p_nonzero))
    elif base == np.e:
        return -np.sum(p_nonzero * np.log(p_nonzero))
    else:
        return -np.sum(p_nonzero * np.log(p_nonzero)) / np.log(base)

# Test cases
print("Testing entropy function:")
print(f"  Fair coin [0.5, 0.5]: H = {entropy([0.5, 0.5]):.4f} bits (expected: 1)")
print(f"  Fair 8-sided die: H = {entropy([1/8]*8):.4f} bits (expected: 3)")
print(f"  Biased [0.9, 0.1]: H = {entropy([0.9, 0.1]):.4f} bits (expected: ~0.47)")
print(f"  Deterministic [1, 0]: H = {entropy([1, 0]):.4f} bits (expected: 0)")
print(f"  Uniform over 4: H = {entropy([0.25]*4):.4f} bits (expected: 2)")
```

---

### Solution B2 — Entropy vs Bias Plot

```python
import numpy as np
import matplotlib.pyplot as plt

def binary_entropy(p):
    """Entropy of Bernoulli(p) distribution."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

p_vals = np.linspace(0.001, 0.999, 200)
H_vals = [binary_entropy(p) for p in p_vals]

plt.figure(figsize=(10, 6))
plt.plot(p_vals, H_vals, 'b-', linewidth=2)
plt.axhline(y=1, color='red', linestyle='--', label='Maximum H = 1 bit')
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

# Mark p = 0.9
p_biased = 0.9
H_biased = binary_entropy(p_biased)
plt.plot(p_biased, H_biased, 'ro', markersize=10)
plt.annotate(f'p=0.9: H={H_biased:.3f}', xy=(p_biased, H_biased), 
             xytext=(0.75, 0.6), fontsize=12,
             arrowprops=dict(arrowstyle='->', color='red'))

plt.xlabel('P(X = 1)', fontsize=12)
plt.ylabel('Entropy H(X) [bits]', fontsize=12)
plt.title('Binary Entropy Function', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.show()

print(f"Maximum entropy at p = 0.5: H = {binary_entropy(0.5):.4f} bits")
print(f"Entropy at p = 0.9: H = {binary_entropy(0.9):.4f} bits")
```

---

### Solution B3 — Huffman Coding Implementation

```python
import heapq
from collections import Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    """Build Huffman tree from frequency dictionary."""
    heap = [HuffmanNode(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    return heap[0]

def generate_codes(node, code="", codes=None):
    """Generate Huffman codes by traversing tree."""
    if codes is None:
        codes = {}
    
    if node.char is not None:
        codes[node.char] = code if code else "0"
    else:
        if node.left:
            generate_codes(node.left, code + "0", codes)
        if node.right:
            generate_codes(node.right, code + "1", codes)
    
    return codes

def huffman_encode(text, codes):
    """Encode text using Huffman codes."""
    return ''.join(codes[char] for char in text)

def huffman_decode(encoded, root):
    """Decode Huffman-encoded string."""
    decoded = []
    node = root
    
    for bit in encoded:
        node = node.left if bit == '0' else node.right
        if node.char is not None:
            decoded.append(node.char)
            node = root
    
    return ''.join(decoded)

# Example usage
text = "abracadabra"
freq = Counter(text)
total = sum(freq.values())
probs = {char: count/total for char, count in freq.items()}

# Build tree and generate codes
tree = build_huffman_tree(freq)
codes = generate_codes(tree)

# Calculate entropy and average code length
entropy_val = -sum(p * np.log2(p) for p in probs.values())
avg_code_length = sum(probs[char] * len(codes[char]) for char in probs)

print(f"Text: '{text}'")
print(f"\nCharacter frequencies: {dict(freq)}")
print(f"Character probabilities: {probs}")
print(f"\nHuffman codes:")
for char, code in sorted(codes.items()):
    print(f"  '{char}': {code} (length {len(code)})")

print(f"\nEntropy: {entropy_val:.4f} bits/symbol")
print(f"Average code length: {avg_code_length:.4f} bits/symbol")
print(f"Efficiency: {entropy_val/avg_code_length:.2%}")

# Encode and decode
encoded = huffman_encode(text, codes)
decoded = huffman_decode(encoded, tree)
print(f"\nEncoded: {encoded}")
print(f"Decoded: {decoded}")
print(f"Correct: {decoded == text}")
print(f"Compression: {len(text)*8} bits → {len(encoded)} bits ({len(encoded)/(len(text)*8):.1%})")
```

---

### Solution B4 — Empirical Entropy Estimation

```python
import numpy as np
import matplotlib.pyplot as plt

def estimate_entropy_from_samples(samples, num_bins=None):
    """Estimate entropy from samples using histogram."""
    if num_bins is None:
        num_bins = len(np.unique(samples))
    
    counts, _ = np.histogram(samples, bins=num_bins)
    probs = counts / len(samples)
    probs = probs[probs > 0]  # Remove zeros
    
    return -np.sum(probs * np.log2(probs))

# True distribution: Categorical with known probabilities
true_probs = np.array([0.4, 0.3, 0.2, 0.1])
true_entropy = -np.sum(true_probs * np.log2(true_probs))

# Estimate entropy with increasing sample sizes
sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
estimates = []
std_devs = []

n_trials = 50

for n in sample_sizes:
    trial_estimates = []
    for _ in range(n_trials):
        samples = np.random.choice(len(true_probs), size=n, p=true_probs)
        est = estimate_entropy_from_samples(samples, num_bins=len(true_probs))
        trial_estimates.append(est)
    estimates.append(np.mean(trial_estimates))
    std_devs.append(np.std(trial_estimates))

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(sample_sizes, estimates, yerr=std_devs, fmt='o-', capsize=5, label='Estimated H')
plt.axhline(y=true_entropy, color='red', linestyle='--', linewidth=2, label=f'True H = {true_entropy:.4f}')
plt.xscale('log')
plt.xlabel('Sample Size', fontsize=12)
plt.ylabel('Entropy (bits)', fontsize=12)
plt.title('Entropy Estimation Convergence', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

print(f"True entropy: {true_entropy:.4f} bits")
print(f"\nSample Size | Estimated H | Error")
print("-" * 40)
for n, est in zip(sample_sizes, estimates):
    print(f"{n:>10} | {est:>11.4f} | {abs(est - true_entropy):>6.4f}")
```

---

### Solution B5 — Cross-Entropy Loss in Classification

```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """Numerically stable softmax."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """Cross-entropy loss for classification."""
    n = len(y_true)
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Generate toy data: 2D points, 3 classes
np.random.seed(42)
n_samples = 300
n_classes = 3

# Generate data in clusters
X = np.vstack([
    np.random.randn(100, 2) + [0, 2],
    np.random.randn(100, 2) + [2, -1],
    np.random.randn(100, 2) + [-2, -1]
])
y_labels = np.array([0]*100 + [1]*100 + [2]*100)
y_onehot = np.eye(n_classes)[y_labels]

# Initialize weights
W = np.random.randn(2, n_classes) * 0.01
b = np.zeros(n_classes)

# Training with gradient descent
lr = 0.1
losses = []

for epoch in range(200):
    # Forward pass
    logits = X @ W + b
    probs = softmax(logits)
    
    # Loss
    loss = cross_entropy_loss(y_onehot, probs)
    losses.append(loss)
    
    # Backward pass (gradient)
    grad_logits = (probs - y_onehot) / n_samples
    grad_W = X.T @ grad_logits
    grad_b = np.sum(grad_logits, axis=0)
    
    # Update
    W -= lr * grad_W
    b -= lr * grad_b

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(losses, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12)
axes[0].set_title('Training Loss', fontsize=14)

# Decision boundary
xx, yy = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-4, 5, 100))
Z = softmax(np.c_[xx.ravel(), yy.ravel()] @ W + b)
Z = np.argmax(Z, axis=1).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='Set3')
axes[1].scatter(X[:, 0], X[:, 1], c=y_labels, cmap='Set1', edgecolors='black', s=30)
axes[1].set_xlabel('$x_1$', fontsize=12)
axes[1].set_ylabel('$x_2$', fontsize=12)
axes[1].set_title('Classification Result', fontsize=14)

plt.tight_layout()
plt.show()

# Final accuracy
predictions = np.argmax(softmax(X @ W + b), axis=1)
accuracy = np.mean(predictions == y_labels)
print(f"Final loss: {losses[-1]:.4f}")
print(f"Final accuracy: {accuracy:.2%}")
```

---

## Part C: Conceptual Solutions

### Solution C1
Entropy measures average "surprise" or uncertainty. Uniform distributions maximize uncertainty because all outcomes are equally likely — you have no information to predict which will occur. Any non-uniform distribution gives you some predictive power (expect more likely outcomes), reducing uncertainty.

### Solution C2
If $H(X) = 0$, the distribution is **deterministic** — one outcome has probability 1, all others have probability 0. Zero entropy means zero uncertainty, which only happens when we know exactly what will occur.

### Solution C3
**Discrete entropy** measures absolute uncertainty. Since $-p_i \log p_i \geq 0$ for all valid $p_i$, the sum is always non-negative.

**Differential entropy** measures uncertainty relative to a reference (uniform density). It can be negative when the distribution is more concentrated than the reference. For example, a very narrow Gaussian has negative entropy because it's more "certain" than the implicit uniform reference.

### Solution C4
Cross-entropy loss: $L = -\sum_i y_i \log \hat{y}_i$

For one-hot labels (classification), this equals $-\log \hat{y}_c$ where $c$ is the correct class.

**MLE connection**: If we model $P(y|x) = \hat{y}$, the likelihood of dataset $\{(x_i, y_i)\}$ is:
$$L(\theta) = \prod_i P(y_i|x_i; \theta)$$

Log-likelihood:
$$\log L = \sum_i \log P(y_i|x_i) = \sum_i \log \hat{y}_{i,c_i}$$

**Maximizing log-likelihood = Minimizing negative log-likelihood = Minimizing cross-entropy!**
