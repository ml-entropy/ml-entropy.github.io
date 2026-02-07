# Tutorial 14: Entropy, Cross-Entropy, and KL Divergence — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Definitions
a) **Entropy $H(P)$**: The average amount of information or "surprise" from a distribution $P$. It's the theoretical minimum average number of bits required to encode a message from $P$.
b) **Cross-Entropy $H(P, Q)$**: The average number of bits required to encode messages from a true distribution $P$ when using a code optimized for a different distribution $Q$.
c) **KL Divergence $D_{KL}(P || Q)$**: The "penalty" or extra bits wasted by using a code from $Q$ instead of the optimal code from $P$. It measures the inefficiency of the approximation $Q$.

**Relationship**: KL Divergence is the difference between Cross-Entropy and Entropy. $D_{KL}(P || Q) = H(P, Q) - H(P)$.

### Solution A2 — The Grand Equation
The equation is:
$$ H(P, Q) = H(P) + D_{KL}(P || Q) $$

In machine learning, we want to make our model's distribution $Q$ as close as possible to the true data distribution $P$. This is equivalent to minimizing the "distance" between them, which is the KL Divergence $D_{KL}(P || Q)$.

When we train a model, the true data distribution $P$ is fixed, which means its entropy $H(P)$ is a constant. Therefore, minimizing the cross-entropy $H(P, Q)$ has the exact same effect as minimizing the KL divergence, because the constant $H(P)$ does not affect the location of the minimum.

### Solution A3 — Weather Prediction
Given $P = [0.5, 0.5]$ and $Q = [0.8, 0.2]$.

a) **Entropy**:
$$ H(P) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = - (0.5 \times -1 + 0.5 \times -1) = \boxed{1 \text{ bit}} $$

b) **Cross-Entropy**:
$$ H(P, Q) = - (0.5 \log_2 0.8 + 0.5 \log_2 0.2) = - (0.5 \times -0.322 + 0.5 \times -2.322) = -(-0.161 - 1.161) = \boxed{1.322 \text{ bits}} $$

c) **KL Divergence**:
$$ D_{KL}(P || Q) = \sum P(x) \log_2 \frac{P(x)}{Q(x)} = 0.5 \log_2 \frac{0.5}{0.8} + 0.5 \log_2 \frac{0.5}{0.2} $$
$$ = 0.5 \log_2(0.625) + 0.5 \log_2(2.5) = 0.5 \times (-0.678) + 0.5 \times (1.322) = -0.339 + 0.661 = \boxed{0.322 \text{ bits}} $$

d) **Verification**:
$$ H(P) + D_{KL}(P || Q) = 1 + 0.322 = 1.322 = H(P, Q) $$
The equation holds.

### Solution A4 — Perfect vs. Terrible Model
Given $P = [1, 0, 0]$.

a) **Entropy**:
$$ H(P) = - (1 \log_2 1 + 0 \log_2 0 + 0 \log_2 0) = 0 $$
(Note: $0 \log 0$ is defined as 0). There is no uncertainty, so entropy is zero.

b) **Perfect Model**: $Q_{perfect} = [0.99, 0.005, 0.005]$
$$ H(P, Q_{perfect}) = - (1 \log_2 0.99 + 0 \log_2 0.005 + 0 \log_2 0.005) = - \log_2 0.99 \approx \boxed{0.014 \text{ bits}} $$
This is a very low loss.

c) **Terrible Model**: $Q_{terrible} = [0.1, 0.5, 0.4]$
$$ H(P, Q_{terrible}) = - (1 \log_2 0.1 + 0 \log_2 0.5 + 0 \log_2 0.4) = - \log_2 0.1 \approx \boxed{3.32 \text{ bits}} $$
This is a much higher loss.

d) **Comparison**: Cross-entropy heavily penalizes models that are "confidently wrong." The terrible model assigned a very low probability (0.1) to the true class, resulting in a high loss. The perfect model was very confident about the correct class, resulting in a loss close to zero.

### Solution A5 — Asymmetry of KL Divergence
$$ D_{KL}(Q || P) = \sum Q(x) \log_2 \frac{Q(x)}{P(x)} = 0.8 \log_2 \frac{0.8}{0.5} + 0.2 \log_2 \frac{0.2}{0.5} $$
$$ = 0.8 \log_2(1.6) + 0.2 \log_2(0.4) = 0.8 \times (0.678) + 0.2 \times (-1.322) = 0.542 - 0.264 = \boxed{0.278 \text{ bits}} $$
This is not equal to $D_{KL}(P || Q) = 0.322$. KL Divergence is asymmetric.

**Importance**: This reflects that the "cost" of approximating P with Q is different from the cost of approximating Q with P. For example, if P has zero probability for an event but Q gives it a non-zero probability, $D_{KL}(P || Q)$ is fine, but $D_{KL}(Q || P)$ would be infinite. This is crucial in generative models where we want our model Q to avoid generating samples that are impossible under P.

---

## Part B: Coding Solutions

### Solution B1 — Basic Calculator
```python
import numpy as np

def entropy(P):
    """Calculate Shannon Entropy."""
    # Add a small epsilon to prevent log(0)
    P = P + 1e-9
    return -np.sum(P * np.log2(P))

def cross_entropy(P, Q):
    """Calculate Cross-Entropy."""
    P = P + 1e-9
    Q = Q + 1e-9
    return -np.sum(P * np.log2(Q))

def kl_divergence(P, Q):
    """Calculate KL Divergence."""
    P = P + 1e-9
    Q = Q + 1e-9
    return np.sum(P * np.log2(P / Q))

# Example from A3
P = np.array([0.5, 0.5])
Q = np.array([0.8, 0.2])

H_P = entropy(P)
H_PQ = cross_entropy(P, Q)
D_KL_PQ = kl_divergence(P, Q)

print(f"H(P) = {H_P:.3f}")
print(f"H(P, Q) = {H_PQ:.3f}")
print(f"D_KL(P || Q) = {D_KL_PQ:.3f}")
print(f"H(P) + D_KL(P || Q) = {H_P + D_KL_PQ:.3f}")
```

### Solution B2 — Visualizing the Loss
```python
import matplotlib.pyplot as plt

# True label is class 0
P = np.array([1.0, 0.0])

# Model's predicted probability for the correct class
p_correct = np.linspace(0.01, 0.99, 100)

# Calculate cross-entropy for each prediction
losses = []
for p in p_correct:
    Q = np.array([p, 1-p])
    loss = cross_entropy(P, Q)
    losses.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(p_correct, losses, lw=2)
plt.xlabel("Predicted Probability for Correct Class")
plt.ylabel("Cross-Entropy Loss")
plt.title("Cross-Entropy Loss vs. Model Confidence")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```
**Interpretation**: The curve shows that the loss is very close to zero when the model correctly predicts a high probability for the true class. However, as the predicted probability for the true class approaches zero, the loss skyrockets towards infinity. This demonstrates that cross-entropy severely punishes predictions that are confidently wrong.

---

## Part C: Conceptual Solutions

### Solution C1 — Why Not MSE?
1.  **Non-convex Loss Surface**: For models that use a sigmoid or softmax output, using MSE can create a non-convex loss surface with multiple local minima, making optimization difficult. Cross-entropy provides a convex loss surface, which is much easier to optimize.
2.  **Gradient Saturation**: When a prediction is very wrong (e.g., sigmoid output is 0 when it should be 1), the gradient of the sigmoid function becomes very small. With MSE, this small gradient leads to very slow learning. The derivative of cross-entropy with a sigmoid/softmax output cancels out the sigmoid's derivative term, resulting in a strong gradient (`p - y`) that learns quickly from large errors.

### Solution C2 — VAEs and KL Divergence
The KL divergence term in a VAE loss function, $D_{KL}(q(z|x) || p(z))$, acts as a **regularizer** on the latent space.

-   **What it enforces**: It forces the distribution of the learned latent vectors (the encodings), $q(z|x)$, to be close to a prior distribution, typically a standard normal distribution $\mathcal{N}(0, 1)$. This means the model is penalized if it creates encodings that are "far" from a simple, continuous, and well-defined distribution.

-   **What would happen without it**: Without the KL term, the VAE would behave like a standard autoencoder. It would learn to perfectly reconstruct the input by creating highly specific, disjointed encodings for each data point. The latent space would have no regular structure, and you wouldn't be able to sample from it to generate new, plausible data. The KL term ensures the latent space is **continuous and dense**, which is the key property that allows for meaningful generation.

