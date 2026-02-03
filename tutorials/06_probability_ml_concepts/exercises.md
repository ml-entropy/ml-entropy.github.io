# Tutorial 06: Probability Concepts in ML — Exercises

## Difficulty Levels
- 🟢 Easy | 🟡 Medium | 🔴 Hard

---

## Part A: Theory & Derivation

### Exercise A1 🟢 — Joint and Marginal
Given joint probability table:

|     | Y=0 | Y=1 |
|-----|-----|-----|
| X=0 | 0.2 | 0.1 |
| X=1 | 0.3 | 0.4 |

a) Find $P(X=1)$
b) Find $P(Y=0|X=1)$
c) Are X and Y independent?

### Exercise A2 🟢 — Conditional Probability
Derive Bayes' theorem from the definition of conditional probability:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### Exercise A3 🟡 — Law of Total Probability
Prove: If $\{B_1, B_2, ..., B_n\}$ partition the sample space, then:
$$P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)$$

### Exercise A4 🟡 — Probability vs Likelihood
A coin has unknown bias $\theta$. You flip it 3 times and get HHT.
a) What is the probability of this outcome given $\theta = 0.6$?
b) What is the likelihood function $L(\theta)$?
c) Find the MLE for $\theta$.

### Exercise A5 🔴 — Bayesian Inference
You have a prior $P(\theta) = \text{Beta}(2, 2)$ for coin bias. After observing 7 heads in 10 flips:
a) Derive the posterior distribution
b) Calculate posterior mean
c) Compare to MLE

### Exercise A6 🔴 — Conjugate Priors
Prove that Beta is conjugate to Binomial: If prior is $\text{Beta}(\alpha, \beta)$ and likelihood is $\text{Binomial}(n, \theta)$, show posterior is $\text{Beta}(\alpha + k, \beta + n - k)$.

---

## Part B: Coding

### Exercise B1 🟢 — Bayes' Theorem Application
```python
# TODO: Medical test problem
# - Disease prevalence: 1%
# - Test sensitivity (true positive rate): 95%
# - Test specificity (true negative rate): 90%
# Calculate: P(disease | positive test)
```

### Exercise B2 🟡 — MLE for Gaussian
```python
# TODO: Implement MLE for Gaussian parameters
# 1. Generate samples from known Gaussian
# 2. Derive and implement MLE formulas
# 3. Compare estimates to true parameters
# 4. Show consistency as n increases
```

### Exercise B3 🔴 — Bayesian vs Frequentist
```python
# TODO: Compare Bayesian and Frequentist inference
# 1. Generate coin flip data from true θ = 0.3
# 2. Compute MLE after each flip
# 3. Compute Bayesian posterior (with Beta(1,1) prior) after each flip
# 4. Plot both estimates over time
# 5. Compare behavior with small sample sizes
```

---

## Part C: Conceptual

### C1 🟡
What's the difference between $P(\text{data}|\theta)$ and $P(\theta|\text{data})$? When is each used?

### C2 🔴
Why is the "prior" controversial in Bayesian statistics? When does the choice of prior matter most?
