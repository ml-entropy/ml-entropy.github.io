# Tutorial 06: Probability Concepts in ML — Solutions

---

## Part A: Theory Solutions

### Solution A1 — Joint and Marginal

a) $P(X=1) = P(X=1, Y=0) + P(X=1, Y=1) = 0.3 + 0.4 = \boxed{0.7}$

b) $P(Y=0|X=1) = \frac{P(Y=0, X=1)}{P(X=1)} = \frac{0.3}{0.7} = \boxed{\frac{3}{7} \approx 0.429}$

c) Check independence: $P(X=1)P(Y=0) = 0.7 \times 0.5 = 0.35 \neq 0.3 = P(X=1, Y=0)$

**X and Y are NOT independent.** ∎

---

### Solution A2 — Deriving Bayes' Theorem

From definition of conditional probability:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
$$P(B|A) = \frac{P(A \cap B)}{P(A)}$$

From the second equation: $P(A \cap B) = P(B|A)P(A)$

Substitute into the first:
$$\boxed{P(A|B) = \frac{P(B|A)P(A)}{P(B)}}$$ ∎

---

### Solution A3 — Law of Total Probability

Since $\{B_i\}$ partition the sample space:
- $A = A \cap \Omega = A \cap (B_1 \cup B_2 \cup ... \cup B_n)$
- $A = (A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n)$

Since $B_i$ are disjoint, so are $(A \cap B_i)$:
$$P(A) = \sum_{i=1}^n P(A \cap B_i)$$

Using $P(A \cap B_i) = P(A|B_i)P(B_i)$:
$$\boxed{P(A) = \sum_{i=1}^n P(A|B_i)P(B_i)}$$ ∎

---

### Solution A4 — Probability vs Likelihood

a) **Probability** of HHT given $\theta = 0.6$:
$$P(\text{HHT}|\theta=0.6) = 0.6 \times 0.6 \times 0.4 = 0.144$$

b) **Likelihood** as function of $\theta$:
$$L(\theta) = P(\text{HHT}|\theta) = \theta^2(1-\theta)$$

c) **MLE**: Maximize $L(\theta)$ or equivalently $\log L(\theta) = 2\log\theta + \log(1-\theta)$
$$\frac{d\log L}{d\theta} = \frac{2}{\theta} - \frac{1}{1-\theta} = 0$$
$$2(1-\theta) = \theta$$
$$\boxed{\hat{\theta}_{MLE} = \frac{2}{3}}$$

---

### Solution A5 — Bayesian Inference

Prior: $P(\theta) = \text{Beta}(2, 2) \propto \theta^1(1-\theta)^1$

Likelihood: $P(D|\theta) = \binom{10}{7}\theta^7(1-\theta)^3 \propto \theta^7(1-\theta)^3$

Posterior (Bayes):
$$P(\theta|D) \propto P(D|\theta)P(\theta) \propto \theta^7(1-\theta)^3 \cdot \theta^1(1-\theta)^1 = \theta^8(1-\theta)^4$$

This is $\boxed{\text{Beta}(9, 5)}$.

**Posterior mean**: $\frac{\alpha}{\alpha + \beta} = \frac{9}{14} \approx 0.643$

**MLE**: $\frac{7}{10} = 0.7$

The Bayesian estimate is pulled toward 0.5 by the prior (regularization effect).

---

### Solution A6 — Conjugate Prior Proof

Prior: $P(\theta) = \text{Beta}(\alpha, \beta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$

Likelihood: $P(D|\theta) = \binom{n}{k}\theta^k(1-\theta)^{n-k}$

Posterior:
$$P(\theta|D) \propto P(D|\theta)P(\theta)$$
$$\propto \theta^k(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}$$
$$= \theta^{k+\alpha-1}(1-\theta)^{(n-k)+\beta-1}$$

This is the kernel of $\text{Beta}(k + \alpha, n - k + \beta)$.

$$\boxed{P(\theta|D) = \text{Beta}(\alpha + k, \beta + n - k)}$$ ∎

---

## Part B: Coding Solutions

### Solution B1 — Medical Test (Bayes)

```python
# Given:
prevalence = 0.01        # P(disease)
sensitivity = 0.95       # P(positive | disease)
specificity = 0.90       # P(negative | no disease)

# Calculate P(positive)
p_positive_given_disease = sensitivity
p_positive_given_no_disease = 1 - specificity  # false positive rate

p_positive = (p_positive_given_disease * prevalence + 
              p_positive_given_no_disease * (1 - prevalence))

# Bayes' theorem: P(disease | positive)
p_disease_given_positive = (sensitivity * prevalence) / p_positive

print(f"P(disease | positive test) = {p_disease_given_positive:.3f}")
print(f"That's only {p_disease_given_positive*100:.1f}%!")

# Interpretation: Even with a positive test, only ~8.8% chance of disease
# This is because the disease is rare (low prevalence)
```

### Solution B2 — MLE for Gaussian

```python
import numpy as np
import matplotlib.pyplot as plt

# True parameters
mu_true = 5.0
sigma_true = 2.0

# MLE formulas (derived from maximizing log-likelihood):
# μ̂ = (1/n) Σ xᵢ
# σ̂² = (1/n) Σ (xᵢ - μ̂)²

np.random.seed(42)
sample_sizes = [5, 10, 50, 100, 500, 1000, 5000]
n_trials = 1000

mu_estimates = []
sigma_estimates = []

for n in sample_sizes:
    mu_trial = []
    sigma_trial = []
    
    for _ in range(n_trials):
        samples = np.random.normal(mu_true, sigma_true, n)
        mu_hat = np.mean(samples)  # MLE for mean
        sigma_hat = np.std(samples, ddof=0)  # MLE for std (biased)
        
        mu_trial.append(mu_hat)
        sigma_trial.append(sigma_hat)
    
    mu_estimates.append((np.mean(mu_trial), np.std(mu_trial)))
    sigma_estimates.append((np.mean(sigma_trial), np.std(sigma_trial)))

# Display results
print("Sample Size | μ̂ (mean ± std) | σ̂ (mean ± std)")
print("-" * 50)
for n, (mu_est, sigma_est) in zip(sample_sizes, zip(mu_estimates, sigma_estimates)):
    print(f"{n:5d}       | {mu_est[0]:.3f} ± {mu_est[1]:.3f}  | {sigma_est[0]:.3f} ± {sigma_est[1]:.3f}")

print(f"\nTrue values: μ = {mu_true}, σ = {sigma_true}")
print("Note: MLE is consistent - estimates converge to true values as n → ∞")
```

### Solution B3 — Bayesian vs Frequentist

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# True parameter
theta_true = 0.3
n_flips = 50

# Generate data
flips = np.random.binomial(1, theta_true, n_flips)

# Track estimates over time
mle_estimates = []
bayesian_means = []
bayesian_stds = []

# Prior: Beta(1, 1) = Uniform
alpha_prior, beta_prior = 1, 1

for i in range(1, n_flips + 1):
    data = flips[:i]
    heads = data.sum()
    tails = i - heads
    
    # MLE: heads / total
    mle = heads / i if i > 0 else 0.5
    mle_estimates.append(mle)
    
    # Bayesian posterior: Beta(alpha + heads, beta + tails)
    alpha_post = alpha_prior + heads
    beta_post = beta_prior + tails
    
    # Posterior mean and std
    post_mean = alpha_post / (alpha_post + beta_post)
    post_std = np.sqrt(alpha_post * beta_post / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)))
    
    bayesian_means.append(post_mean)
    bayesian_stds.append(post_std)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

x = range(1, n_flips + 1)
ax.axhline(theta_true, color='green', linestyle='--', label=f'True θ = {theta_true}')
ax.plot(x, mle_estimates, 'b-', label='MLE', linewidth=2)
ax.plot(x, bayesian_means, 'r-', label='Bayesian posterior mean', linewidth=2)
ax.fill_between(x, 
                np.array(bayesian_means) - 2*np.array(bayesian_stds),
                np.array(bayesian_means) + 2*np.array(bayesian_stds),
                alpha=0.2, color='red', label='95% credible interval')

ax.set_xlabel('Number of flips')
ax.set_ylabel('Estimated θ')
ax.set_title('Bayesian vs Frequentist Estimation')
ax.legend()
ax.set_ylim(0, 1)
plt.show()

print("Key observations:")
print("- MLE can be extreme early on (e.g., 0 or 1 after few flips)")
print("- Bayesian estimate is regularized by prior")
print("- Both converge to true value as n increases")
print("- Bayesian provides uncertainty quantification (credible intervals)")
```

---

## Part C: Conceptual Solutions

### C1

**$P(\text{data}|\theta)$** — Likelihood
- Probability of observing the data given a specific parameter value
- Parameter is fixed, data is random variable
- Used in: MLE, hypothesis testing

**$P(\theta|\text{data})$** — Posterior
- Probability distribution over parameters given observed data
- Data is fixed, parameter is random variable
- Used in: Bayesian inference

Key insight: In frequentist statistics, $\theta$ is a fixed unknown. In Bayesian statistics, $\theta$ is a random variable with a distribution.

### C2

The prior is controversial because:
1. **Subjectivity**: Different people may choose different priors
2. **Influence on results**: With small data, prior dominates the posterior
3. **Philosophical debate**: Frequentists reject treating parameters as random

**When prior matters most**:
- Small sample sizes
- High-dimensional problems
- Sparse data
- Informative priors that conflict with data

**When prior matters least**:
- Large sample sizes (likelihood dominates)
- Uninformative/vague priors
- Data strongly constrains the parameter

In practice, sensitivity analysis (trying multiple priors) helps assess robustness.
