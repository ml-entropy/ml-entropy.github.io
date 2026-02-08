# Tutorial 15: Cross-Entropy — The Cost of Wrong Predictions

## Introduction

Cross-Entropy is arguably the most important loss function for classification problems in machine learning. While it's often presented as a "black box" function to minimize, its origins in information theory reveal a deep and intuitive meaning: **Cross-Entropy measures the average cost of using a wrong model to describe reality.**

This tutorial will build on the concepts of Entropy and KL Divergence to explain what Cross-Entropy is, why it works, and how it's derived.

---

## Part 1: Mathematical Definition

Formally, let $P$ and $Q$ be two probability distributions defined over the same set of events $\mathcal{X}$.

-   $P(x)$ is the **true distribution** (reality).
-   $Q(x)$ is the **estimated distribution** (our model).

The Cross-Entropy $H(P, Q)$ is defined as:

$$ H(P, Q) = - \mathbb{E}_{x \sim P} [\log Q(x)] = - \sum_{x \in \mathcal{X}} P(x) \log Q(x) $$

In the continuous case:

$$ H(P, Q) = - \int P(x) \log Q(x) \, dx $$

### Key Properties

1.  **Non-Negativity**: $H(P, Q) \ge 0$ (assuming discrete events).
2.  **Lower Bound**: $H(P, Q) \ge H(P)$. The cross-entropy is always greater than or equal to the entropy of the true distribution.
    -   equality holds if and only if $P = Q$ almost everywhere.
    -   This is a direct consequence of Gibbs' Inequality.
3.  **Not Symmetric**: $H(P, Q) \neq H(Q, P)$. "The cost of encoding reality $P$ with model $Q$" is not the same as "encoding model $Q$ with reality $P$".

---

## Part 2: The Core Idea — Coding with the Wrong Book

Recall from the Entropy tutorial that the optimal code length for an event with probability $p(x)$ is $-\log_2 p(x)$ bits. The average optimal code length for a distribution $P$ is its entropy, $H(P)$.

Now, imagine two worlds:
1.  **Reality**: The data follows a true, unknown distribution $P$.
2.  **Our Model**: We build a model $Q$ that tries to approximate $P$.

Because we don't know $P$, we are forced to design our compression scheme (our "codebook") based on our model, $Q$. We will assign a code of length $L(x) = -\log_2 Q(x)$ to each event $x$.

What is the average number of bits we will use if we encode events from the real world ($P$) using our codebook designed for our model world ($Q$)?

This is what **Cross-Entropy** calculates.

<div class="figure-container" style="margin: 2rem 0; text-align: center;">
    <svg width="100%" height="220" viewBox="0 0 600 220" style="background: #f8f9fa; border-radius: 8px;">
        <!-- Alice (Source) -->
        <rect x="50" y="60" width="100" height="80" rx="8" fill="#e0e7ff" stroke="#4f46e5" stroke-width="2"/>
        <text x="100" y="95" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#3730a3">True Dist P</text>
        <text x="100" y="115" text-anchor="middle" font-family="Inter" font-size="12" fill="#3730a3">(Reality)</text>
        
        <!-- Arrow 1 -->
        <line x1="150" y1="100" x2="230" y2="100" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"/>
        
        <!-- Codebook Q -->
        <rect x="230" y="40" width="140" height="120" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
        <text x="300" y="70" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#92400e">Model Q</text>
        <text x="300" y="90" text-anchor="middle" font-family="Inter" font-size="12" fill="#92400e">"The Wrong Codebook"</text>
        <text x="300" y="110" text-anchor="middle" font-family="Inter" font-size="11" fill="#92400e">Code Length = -log Q(x)</text>
        <text x="300" y="130" text-anchor="middle" font-family="Inter" font-size="11" fill="#92400e" font-style="italic">Optimized for Q, not P</text>
        
        <!-- Arrow 2 -->
        <line x1="370" y1="100" x2="450" y2="100" stroke="#9ca3af" stroke-width="2" marker-end="url(#arrowhead)"/>
        
        <!-- Result -->
        <rect x="450" y="60" width="120" height="80" rx="8" fill="#fee2e2" stroke="#ef4444" stroke-width="2"/>
        <text x="510" y="95" text-anchor="middle" font-family="Inter" font-weight="bold" fill="#991b1b">Avg Cost</text>
        <text x="510" y="115" text-anchor="middle" font-family="Inter" font-size="12" fill="#991b1b">H(P, Q)</text>
        
        <!-- Definitions -->
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#9ca3af" />
            </marker>
        </defs>
    </svg>
    <p class="caption" style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
        <strong>The Communication Analogy:</strong> Events occur according to Reality ($P$), but we encode them using a scheme based on our Model ($Q$). 
        Since $Q \neq P$, our codes are longer than necessary. Cross-Entropy measures this average length.
    </p>
</div>

$$ H(P, Q) = - \sum_{x \in \mathcal{X}} P(x) \log_2 Q(x) $$

**Key Intuition**:
-   The formula takes the probabilities from the **true distribution** ($P(x)$).
-   ...but it uses the code lengths derived from the **model's distribution** ($-\log_2 Q(x)$).
-   It's the expected number of bits to encode data from reality ($P$) using a code based on our beliefs ($Q$).

---

## Part 3: The Link to KL Divergence

The relationship between Cross-Entropy, Entropy, and KL Divergence is fundamental:

$$ \Large H(P, Q) = H(P) + D_{KL}(P || Q) $$

Let's break this down:

-   **$H(P, Q)$ (Cross-Entropy)**: The average length of our code (based on $Q$) when encoding data from $P$. This is what we actually compute as our loss.
-   **$H(P)$ (Entropy)**: The theoretical minimum average length for any code. This is an irreducible property of the data itself.
-   **$D_{KL}(P || Q)$ (KL Divergence)**: The "penalty" or extra bits we waste because our model $Q$ is not a perfect reflection of reality $P$.

When we train a machine learning model, we are trying to make our model $Q$ as close to the true data distribution $P$ as possible. This is equivalent to minimizing the KL Divergence.

Since the true entropy of the data $H(P)$ is a fixed constant (we can't change the data), **minimizing the Cross-Entropy is mathematically identical to minimizing the KL Divergence.** This is why we use Cross-Entropy as the loss function.

---

## Part 4: The Link to Maximum Likelihood Estimation (MLE)

Why do we use the logarithm? Is it just arbitrary?

No. Minimizing Cross-Entropy is mathematically equivalent to **Maximizing Likelihood**.

Suppose we have a dataset $D = \{(x_1, y_1), \dots, (x_N, y_N)\}$. We want to find the parameters $\theta$ of our model that maximize the probability of seeing this specific dataset.

$$ \mathcal{L}(\theta) = P(Y | X; \theta) = \prod_{i=1}^N P(y_i | x_i; \theta) $$

This product of probabilities becomes very small very quickly. To make it manageable (and turn the product into a sum), we take the log:

$$ \log \mathcal{L}(\theta) = \sum_{i=1}^N \log P(y_i | x_i; \theta) $$

We want to **maximize** this log-likelihood. In machine learning, we prefer to **minimize** a loss function. So we take the negative:

$$ \text{Loss} = - \sum_{i=1}^N \log P(y_i | x_i; \theta) $$

If $y_i$ is the correct class, then $P(y_i | x_i; \theta)$ is exactly what our model $Q$ predicts for that class.
This formula is exactly the **Cross-Entropy** between the empirical distribution (the data) and our model!

---

## Part 5: Cross-Entropy in Classification

Let's see how this applies to a typical multi-class classification problem.

Suppose we have 3 classes: Cat, Dog, Bird. For a single image of a dog, the distributions are:

-   **True Distribution (P)**: This is a one-hot encoded vector. The reality is certain: it's a dog.
    -   $P(\text{Cat}) = 0$
    -   $P(\text{Dog}) = 1$
    -   $P(\text{Bird}) = 0$

-   **Model's Prediction (Q)**: Our model (e.g., after a softmax layer) outputs probabilities.
    -   $Q(\text{Cat}) = 0.1$
    -   $Q(\text{Dog}) = 0.7$
    -   $Q(\text{Bird}) = 0.2$

Now, let's calculate the Cross-Entropy loss for this single data point:

$$ H(P, Q) = - \sum_{i \in \{\text{C,D,B}\}} P(i) \log Q(i) $$
$$ = - [ (0 \cdot \log 0.1) + (1 \cdot \log 0.7) + (0 \cdot \log 0.2) ] $$

Because of the one-hot encoding of $P$, all terms except the one for the true class become zero.

$$ = - \log Q(\text{Dog}) = - \log(0.7) \approx 0.51 $$

This is why "Cross-Entropy loss" in code often simplifies to just the **negative log-likelihood** of the correct class.

---

## Part 6: Comparative Analysis of Loss Functions

Cross-Entropy is the standard, but it's not the only option. How does it compare to others?

### 1. Cross-Entropy vs. Mean Squared Error (MSE)

| Feature | Cross-Entropy (Log Loss) | Mean Squared Error (MSE) |
| :--- | :--- | :--- |
| **Derivation** | Maximum Likelihood (Bernoulli/Multinoulli) | Maximum Likelihood (Gaussian) |
| **Focus** | Probabilities | Regression / Geometric distance |
| **Gradient** | $p - y$ (Strong, constant) | $(p - y)p(1-p)$ (Vanishing) |
| **Penalty** | Exponential (Infinite for confident errors) | Quadratic (Bounded for confident errors) |
| **Best For** | **Classification** | **Regression** |

**Why MSE fails for Classification:**
As shown in the table, the gradient of MSE includes the term $p(1-p)$. When the model is confidently wrong (e.g., $p \approx 0$ for a true label of 1), this term approaches 0. The model stops learning exactly when it needs to learn the most! Cross-Entropy cancels this out, providing a clean gradient of $p - y$.

### 2. Cross-Entropy vs. Hinge Loss (SVM)

-   **Hinge Loss**: $L = \max(0, 1 - y \cdot \hat{y})$ (where $y \in \{-1, 1\}$).
-   **Focus**: Hinge loss cares about the **margin**. Once a point is classified correctly "enough" (outside the margin), the loss is zero. It doesn't care about estimating precise probabilities.
-   **Cross-Entropy**: Cares about the **entire distribution**. Even if correct, it tries to push the probability to exactly 1.0. This makes it better for problems where calibrated probabilities matter.

### 3. Cross-Entropy vs. Focal Loss

-   **Problem**: In imbalanced datasets (e.g., 99% background, 1% object), the model can get low Cross-Entropy by just predicting "background" everywhere.
-   **Focal Loss**: $L = -(1 - p_t)^\gamma \log(p_t)$.
-   It adds a modulating factor $(1 - p_t)^\gamma$ to standard Cross-Entropy.
-   If the model is confident ($p_t \approx 1$), the factor goes to 0, "down-weighting" easy examples.
-   This forces the model to focus on the hard, misclassified examples.

---

## Part 7: Implementation Details (Sparse vs. Categorical)

In frameworks like TensorFlow/Keras or PyTorch, you'll often see two variations:

1.  **Categorical Cross-Entropy**:
    -   Expects labels to be **one-hot encoded** (e.g., `[0, 1, 0]`).
    -   Use this when your targets are vectors.

2.  **Sparse Categorical Cross-Entropy**:
    -   Expects labels to be **integers** (e.g., `1` for Dog).
    -   Use this when your targets are class indices.
    -   **Mathematically identical**, but more memory efficient because you don't need to create the giant one-hot matrix.

---

## Part 8: PyTorch Implementation

In practice, we rarely implement this manually. PyTorch's `nn.CrossEntropyLoss` combines the Softmax and Log-Likelihood steps for numerical stability (using the LogSumExp trick).

```python
import torch
import torch.nn as nn

# 1. Inputs: Raw logits (scores), not probabilities!
# Shape: (Batch Size, Number of Classes)
logits = torch.tensor([[1.5, 2.0, 0.5]])  # Raw scores for [Cat, Dog, Bird]

# 2. True Labels: Class indices (not one-hot vectors)
target = torch.tensor([1])  # Class 1 is "Dog"

# 3. Define and Compute Loss
# This combines nn.LogSoftmax() and nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

print(f"PyTorch Loss: {loss.item():.4f}")

# Verification manually:
# Softmax([1.5, 2.0, 0.5]) -> [0.32, 0.53, 0.12] (approx)
# -log(0.53) -> ~0.63
```

---

## Conclusion

Cross-Entropy is more than just a formula; it's a concept rooted in information theory. It provides a powerful and intuitive way to measure the "cost" of a model's predictions by calculating the average number of bits needed to encode the truth using the model's beliefs.

By minimizing this cost, we are implicitly minimizing the divergence of our model from the true data distribution, leading to effective and efficient training for classification models.

