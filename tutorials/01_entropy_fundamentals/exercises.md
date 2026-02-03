# Tutorial 01: Entropy Fundamentals — Exercises

## Difficulty Levels
- 🟢 Easy: Direct application
- 🟡 Medium: Combining concepts
- 🔴 Hard: Derivations and proofs

---

## Part A: Theory & Derivation Exercises

### Exercise A1 🟢 — Information of Events
Calculate the information content (in bits) for each event:
a) A fair coin lands heads: $P = 0.5$
b) Rolling a 6 on a fair die: $P = 1/6$
c) A certain event: $P = 1$
d) Drawing an ace from a standard deck: $P = 4/52$

---

### Exercise A2 🟢 — Entropy Calculation
Compute the entropy (in bits) of these distributions:
a) Fair coin: $P(H) = P(T) = 0.5$
b) Biased coin: $P(H) = 0.9$, $P(T) = 0.1$
c) Fair 8-sided die
d) Distribution: $P = [0.25, 0.25, 0.25, 0.25]$

---

### Exercise A3 🟡 — Entropy Maximization
a) Among all distributions over 4 outcomes, which has maximum entropy? Prove it.
b) What is the entropy of this maximum-entropy distribution?
c) For $n$ outcomes, what is the maximum entropy?

---

### Exercise A4 🟡 — Cross-Entropy
Given true distribution $P = [0.7, 0.2, 0.1]$ and model $Q = [0.5, 0.3, 0.2]$:
a) Calculate $H(P)$ — entropy of true distribution
b) Calculate $H(P, Q)$ — cross-entropy
c) Calculate $D_{KL}(P || Q)$ — KL divergence
d) Verify: $H(P, Q) = H(P) + D_{KL}(P || Q)$

---

### Exercise A5 🟡 — Joint and Conditional Entropy
For joint distribution:
| | $Y=0$ | $Y=1$ |
|---|-------|-------|
| $X=0$ | 0.4 | 0.1 |
| $X=1$ | 0.1 | 0.4 |

a) Calculate $H(X)$ and $H(Y)$
b) Calculate $H(X, Y)$ — joint entropy
c) Calculate $H(X|Y)$ — conditional entropy
d) Verify: $H(X, Y) = H(Y) + H(X|Y)$

---

### Exercise A6 🔴 — Derive Information Formula
**Derive** that $I(p) = -\log(p)$ from these axioms:
1. $I(p)$ is continuous
2. $I(p) \geq 0$
3. $I(1) = 0$
4. For independent events: $I(p_1 \cdot p_2) = I(p_1) + I(p_2)$

---

### Exercise A7 🔴 — Maximum Entropy Principle
Prove that among all distributions with a given mean $\mu$, the exponential distribution has maximum entropy.

Hint: Use Lagrange multipliers with constraints $\int p(x)dx = 1$ and $\int x p(x)dx = \mu$.

---

### Exercise A8 🔴 — Entropy of Gaussian
Derive the entropy of a Gaussian $N(\mu, \sigma^2)$:
$$h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$$

Show that it depends only on $\sigma$, not $\mu$.

---

## Part B: Coding Exercises

### Exercise B1 🟢 — Entropy Calculator
```python
# TODO: Implement entropy calculation
# 1. Write a function entropy(p) that computes H(X) for distribution p
# 2. Handle edge cases: p=0 (use 0*log(0) = 0)
# 3. Support both log2 (bits) and ln (nats)
# 4. Test on known distributions
```

---

### Exercise B2 🟢 — Entropy vs Bias Plot
```python
# TODO: For a binary distribution P(X=1) = p:
# 1. Plot H(X) as a function of p for p ∈ [0, 1]
# 2. Find the maximum entropy point
# 3. Mark the entropy for p = 0.9 (biased coin)
```

---

### Exercise B3 🟡 — Huffman Coding Implementation
```python
# TODO: Implement Huffman coding
# 1. Build a Huffman tree from symbol frequencies
# 2. Generate codes for each symbol
# 3. Compute average code length
# 4. Compare to entropy (should be close!)
# 5. Encode and decode a sample message
```

---

### Exercise B4 🟡 — Empirical Entropy Estimation
```python
# TODO: Estimate entropy from samples
# 1. Generate samples from a known distribution
# 2. Estimate PMF from histogram
# 3. Compute entropy of estimated PMF
# 4. Compare to true entropy
# 5. Plot convergence as sample size increases
```

---

### Exercise B5 🔴 — Cross-Entropy Loss in Classification
```python
# TODO: Implement cross-entropy loss for classification
# 1. Implement softmax function
# 2. Implement cross-entropy loss
# 3. Generate toy classification data
# 4. Train a simple linear classifier using cross-entropy
# 5. Plot the loss curve during training
```

---

### Exercise B6 🔴 — Differential Entropy Estimation
```python
# TODO: Estimate differential entropy from continuous samples
# 1. Generate samples from N(0, σ²)
# 2. Use k-nearest neighbor entropy estimator
# 3. Compare to analytical h(X) = 0.5*log(2πeσ²)
# 4. Plot estimation error vs sample size
```

---

## Part C: Conceptual Questions

### Exercise C1 🟢
Why is entropy maximized for uniform distributions?

### Exercise C2 🟡
If $H(X) = 0$, what can you say about the distribution of $X$?

### Exercise C3 🟡
Why can differential entropy be negative, but discrete entropy cannot?

### Exercise C4 🔴
Explain why minimizing cross-entropy loss in ML is equivalent to maximum likelihood estimation.
