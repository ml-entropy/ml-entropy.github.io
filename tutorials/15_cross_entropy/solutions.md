# Tutorial 15: Cross-Entropy — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Definition in Your Own Words
From an information theory perspective, Cross-Entropy is the average number of bits required to encode data coming from a true distribution $P$ when we use a code optimized for a different, approximate distribution $Q$.

It's useful for machine learning because it provides a measure of how "bad" our model's predictions ($Q$) are at describing the true data labels ($P$). A high cross-entropy means our model is a poor approximation of reality, resulting in a high loss. Minimizing cross-entropy forces the model's predictions to get closer to the true distribution.

### Solution A2 — The Loss Formula
Let the true label be class $k$. The one-hot vector $y$ will have $y_k = 1$ and $y_i = 0$ for all $i \neq k$.

The Cross-Entropy formula is:
$$ H(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log \hat{y}_i $$
where $C$ is the number of classes.

When we expand the sum, every term where $y_i = 0$ is multiplied by zero, so it disappears:
$$ = - (y_1 \log \hat{y}_1 + \dots + y_k \log \hat{y}_k + \dots + y_C \log \hat{y}_C) $$
$$ = - (0 \cdot \log \hat{y}_1 + \dots + 1 \cdot \log \hat{y}_k + \dots + 0 \cdot \log \hat{y}_C) $$
$$ = - \log \hat{y}_k $$
This shows that the loss simplifies to the negative logarithm of the probability assigned to the single correct class.

### Solution A3 — Manual Calculation
a) **True Distribution P**: The image is a Banana, so the true distribution is a one-hot vector:
$P = \{\text{Apple: } 0, \text{Banana: } 1, \text{Cherry: } 0\}$

b) **Cross-Entropy Loss**:
Using the simplification from A2, the loss is just the negative log of the probability for the correct class (Banana).
$$ \text{Loss} = - \log(Q(\text{Banana})) = - \log(0.5) \approx \boxed{0.693} $$
(Using natural log, which is standard in ML frameworks). If using log base 2, the answer would be 1 bit.

c) **Perfectly Confident and Correct**: If $Q(\text{Banana}) = 1$:
$$ \text{Loss} = - \log(1) = \boxed{0} $$
A perfect prediction results in zero loss.

d) **Perfectly Confident and Incorrect**: If $Q(\text{Apple}) = 1$, then $Q(\text{Banana})$ must be 0 (or a very small number, for stability).
$$ \text{Loss} = - \log(Q(\text{Banana})) = - \log(0) = \boxed{\infty} $$
The loss is infinite, indicating an extremely high penalty for being confidently wrong.

### Solution A4 — Cross-Entropy vs. MSE
a) **BCE Loss**: $L_{BCE} = - (y \log(p) + (1-y) \log(1-p))$. For $y=1$, this is $L_{BCE} = -\log(p)$.
b) **MSE Loss**: $L_{MSE} = (y - p)^2$. For $y=1$, this is $L_{MSE} = (1 - p)^2$.

c) **Gradients**: Let $z$ be the logit, so $p = \sigma(z) = 1 / (1 + e^{-z})$. The derivative of the sigmoid is $\sigma'(z) = \sigma(z)(1-\sigma(z)) = p(1-p)$.

-   **BCE Gradient**:
    $$ \frac{\partial L_{BCE}}{\partial z} = \frac{\partial L_{BCE}}{\partial p} \frac{\partial p}{\partial z} = \left(-\frac{1}{p}\right) \cdot (p(1-p)) = -(1-p) = p - 1 $$
-   **MSE Gradient**:
    $$ \frac{\partial L_{MSE}}{\partial z} = \frac{\partial L_{MSE}}{\partial p} \frac{\partial p}{\partial z} = (-2(1-p)) \cdot (p(1-p)) = -2(1-p)^2 p $$

d) **Why BCE is better**: Look at the gradients when the model is confidently wrong. If the true label is 1, a confidently wrong prediction means $p \to 0$.
-   **BCE Gradient**: As $p \to 0$, the gradient $\frac{\partial L_{BCE}}{\partial z} \to -1$. It's a strong, constant signal to update the weights.
-   **MSE Gradient**: As $p \to 0$, the gradient $\frac{\partial L_{MSE}}{\partial z} \to 0$. The gradient vanishes! The model learns extremely slowly when it needs to learn the most. This is known as the "vanishing gradient" problem for MSE in classification.

---

## Part B: Coding Solutions

### Solution B1 — Cross-Entropy Function
```python
import numpy as np

def cross_entropy(P, Q):
    """
    Calculates the cross-entropy between two probability distributions.
    
    Args:
        P (np.array): The true distribution (one-hot).
        Q (np.array): The predicted distribution.
        
    Returns:
        float: The cross-entropy loss.
    """
    # Add a small epsilon for numerical stability to prevent log(0)
    epsilon = 1e-9
    Q = np.clip(Q, epsilon, 1. - epsilon)
    
    return -np.sum(P * np.log(Q))

# Verification from A3
P_banana = np.array([0, 1, 0])
Q_banana = np.array([0.2, 0.5, 0.3])

loss = cross_entropy(P_banana, Q_banana)
print(f"Calculated Loss: {loss:.3f}")
print(f"Expected Loss: {-np.log(0.5):.3f}")

# What happens if Q has a zero?
Q_zero = np.array([0.2, 0, 0.8])
# Without epsilon, this would cause a `RuntimeWarning: divide by zero in log`
# and result in `inf`. Our stable function handles it.
loss_stable = cross_entropy(P_banana, Q_zero)
print(f"Loss with a zero prediction (stable): {loss_stable:.3f}")
```

### Solution B2 — Visualizing the Loss
```python
import matplotlib.pyplot as plt

# Predicted probabilities for the correct class (y=1)
p = np.linspace(0.01, 0.99, 100)

# BCE Loss for y=1 is -log(p)
bce_loss = -np.log(p)

# MSE Loss for y=1 is (1-p)^2
mse_loss = (1 - p)**2

plt.figure(figsize=(10, 6))
plt.plot(p, bce_loss, label='Cross-Entropy Loss', lw=2)
plt.plot(p, mse_loss, label='Mean Squared Error Loss', lw=2)
plt.xlabel("Predicted Probability for Correct Class (p)")
plt.ylabel("Loss")
plt.title("BCE vs. MSE Loss for a Correct Label of 1")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```
**Interpretation**: The visualization clearly shows that as the prediction `p` gets confidently wrong (approaches 0), the Cross-Entropy loss shoots up towards infinity, creating a powerful gradient. The MSE loss, in contrast, flattens out, leading to a vanishing gradient and slow learning.

---

## Part C: Conceptual Solutions

### Solution C1 — Label Smoothing
a) **How it changes the loss**: Instead of having only one non-zero term in the cross-entropy sum, all terms will be non-zero. The model is now penalized for being too confident even in its correct predictions, as it must also assign some probability mass to the other classes. For $y_{smooth} = [0.05, 0.9, 0.05]$ and $\hat{y} = [0.1, 0.8, 0.1]$, the loss is now:
`-(0.05*log(0.1) + 0.9*log(0.8) + 0.05*log(0.1))` instead of just `-log(0.8)`.

b) **Goal**: The goal is to prevent the model from becoming overconfident. By forcing it to be "less sure" (i.e., never predicting a hard 0 or 1), it creates a softer decision boundary and can improve generalization and calibration. It regularizes the model by adding noise to the labels.

c) **Relationship to KL Divergence**: Minimizing the cross-entropy $H(y_{smooth}, \hat{y})$ is equivalent to minimizing the KL divergence $D_{KL}(y_{smooth} || \hat{y})$. Label smoothing, therefore, encourages the model's output distribution $\hat{y}$ to stay close to the smoothed target distribution, not a one-hot distribution.

### Solution C2 — The Role of the Logarithm
-   **Is it a valid loss?**: Yes, "Linear-Entropy" or $-\sum P(x) Q(x)$ would be a valid loss function in the sense that it is minimized when $Q(x)$ is 1 for the correct class (assuming $P$ is one-hot). This is related to the dot product, maximizing the alignment between the prediction and true vectors.

-   **What is lost?**: The most critical property of the logarithm is lost: **the infinite penalty for confident failure**.
    -   With $-\log Q(x)$, if $Q(x) \to 0$, the loss $\to \infty$.
    -   With $-Q(x)$, if $Q(x) \to 0$, the loss $\to 0$.
    This is the opposite of what we want! A linear loss would not heavily penalize a model for being confidently wrong. The gradients would be small and constant, lacking the powerful error signal that makes cross-entropy so effective. The logarithmic scale correctly maps the multiplicative nature of probabilities to an additive, unbounded scale of "surprise" or "cost".

