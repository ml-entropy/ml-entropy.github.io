# Tutorial 2: KL Divergence

## 🎯 The Big Picture

**KL Divergence measures how many extra bits you waste by using the wrong probability distribution.**

If you think data comes from $Q$ but it actually comes from $P$, the KL divergence $D_{KL}(P \| Q)$ tells you exactly how inefficient your encoding is.

---

## 1. Cross-Entropy First: Setting the Stage

Before deriving KL divergence, we need to understand **cross-entropy**.

### The Coding Problem

Suppose the true distribution of messages is $P$, but we design our code assuming distribution $Q$. How many bits do we need on average?

### Deriving Cross-Entropy (Discrete)

**Step 1:** With distribution $Q$, the optimal code length for symbol $x$ is:
$$\ell_Q(x) = -\log_2 Q(x) \text{ bits}$$

(This comes from Shannon's source coding theorem: you can't do better than entropy, and optimal codes achieve it.)

**Step 2:** But the symbols actually come from $P$, so symbol $x$ appears with probability $P(x)$.

**Step 3:** The **average code length** when using $Q$'s code on data from $P$:
$$H(P, Q) = \sum_x P(x) \cdot \ell_Q(x) = \sum_x P(x) \cdot (-\log_2 Q(x))$$

### Cross-Entropy Formula (Discrete)

$$\boxed{H(P, Q) = -\sum_x P(x) \log Q(x)}$$

**Interpretation:** Average bits needed when using wrong distribution $Q$ to encode data from $P$.

---

## 2. Deriving KL Divergence (Discrete)

### The Waste Calculation

How many **extra** bits do we waste by using $Q$ instead of $P$?

**Optimal average code length** (if we knew $P$):
$$H(P) = -\sum_x P(x) \log P(x)$$

**Actual average code length** (using $Q$):
$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

**Extra bits wasted:**
$$D_{KL}(P \| Q) = H(P, Q) - H(P)$$

### Deriving the Formula

$$D_{KL}(P \| Q) = -\sum_x P(x) \log Q(x) - \left(-\sum_x P(x) \log P(x)\right)$$

$$= -\sum_x P(x) \log Q(x) + \sum_x P(x) \log P(x)$$

$$= \sum_x P(x) \left[\log P(x) - \log Q(x)\right]$$

$$\boxed{D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}}$$

### Alternative Form: Expectation

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

The expectation is over the **true** distribution $P$.

---

## 3. Proving KL Divergence is Non-Negative

This is not obvious! Let's prove $D_{KL}(P \| Q) \geq 0$.

### Using Jensen's Inequality

**Jensen's Inequality:** For a convex function $f$ and random variable $X$:
$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

For concave functions (like $\log$), the inequality reverses:
$$\log(\mathbb{E}[X]) \geq \mathbb{E}[\log(X)]$$

### The Proof

$$-D_{KL}(P \| Q) = -\sum_x P(x) \log \frac{P(x)}{Q(x)} = \sum_x P(x) \log \frac{Q(x)}{P(x)}$$

$$= \mathbb{E}_{x \sim P}\left[\log \frac{Q(x)}{P(x)}\right]$$

By Jensen's inequality (log is concave):
$$\mathbb{E}_{x \sim P}\left[\log \frac{Q(x)}{P(x)}\right] \leq \log \mathbb{E}_{x \sim P}\left[\frac{Q(x)}{P(x)}\right]$$

$$= \log \sum_x P(x) \cdot \frac{Q(x)}{P(x)} = \log \sum_x Q(x) = \log 1 = 0$$

Therefore:
$$-D_{KL}(P \| Q) \leq 0 \implies D_{KL}(P \| Q) \geq 0 \quad \checkmark$$

**Equality holds if and only if** $\frac{Q(x)}{P(x)}$ is constant, which means $P = Q$.

---

## 4. Detailed Example Calculation

Let's compute KL divergence step by step.

**True distribution:** $P = [0.5, 0.3, 0.2]$ (outcomes A, B, C)
**Model distribution:** $Q = [0.33, 0.33, 0.34]$ (approximately uniform)

| $x$ | $P(x)$ | $Q(x)$ | $\frac{P(x)}{Q(x)}$ | $\log_2\frac{P(x)}{Q(x)}$ | $P(x) \log_2\frac{P(x)}{Q(x)}$ |
|-----|--------|--------|---------------------|---------------------------|-------------------------------|
| A | 0.5 | 0.33 | 1.515 | 0.600 | 0.300 |
| B | 0.3 | 0.33 | 0.909 | -0.138 | -0.041 |
| C | 0.2 | 0.34 | 0.588 | -0.765 | -0.153 |
| **Total** | 1.0 | 1.0 | | | **0.106 bits** |

So $D_{KL}(P \| Q) \approx 0.106$ bits.

**Interpretation:** Using $Q$ instead of $P$ wastes about 0.1 bits per symbol on average.

---

## 5. From Discrete to Continuous: KL Divergence

### The Transition (Following Tutorial 01 Pattern)

**Step 1:** Discretize into bins of width $\Delta x$.

Discrete probabilities:
- $p_i = P(\text{bin } i) \approx p(x_i) \Delta x$
- $q_i = Q(\text{bin } i) \approx q(x_i) \Delta x$

**Step 2:** Write discrete KL:
$$D_{KL}(P \| Q) = \sum_i p_i \log \frac{p_i}{q_i}$$

**Step 3:** Substitute:
$$= \sum_i p(x_i) \Delta x \cdot \log \frac{p(x_i) \Delta x}{q(x_i) \Delta x}$$

$$= \sum_i p(x_i) \Delta x \cdot \log \frac{p(x_i)}{q(x_i)}$$

The $\Delta x$ in the ratio **cancels**! This is crucial.

**Step 4:** Take limit as $\Delta x \to 0$:

$$D_{KL}(P \| Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx$$

### Why KL Divergence Works for Continuous Distributions

Unlike differential entropy, KL divergence:
- Is always **non-negative**
- Is **scale-invariant** (the $\Delta x$ cancels)
- Has the same interpretation as discrete case

This is because KL is a **relative** quantity—it compares $P$ to $Q$.

### Continuous KL Formula

$$\boxed{D_{KL}(P \| Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx}$$

---

## 6. The Critical Asymmetry

### Forward KL: $D_{KL}(P \| Q)$

**"Mean-seeking" or "Moment-matching"**

- Expectation is over $P$ (the true distribution)
- Penalizes $Q$ for assigning low probability where $P$ is high
- **Forces $Q$ to cover all modes of $P$**
- Results in overdispersed approximations

$$D_{KL}(P \| Q) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

When $P(x) > 0$ but $Q(x) \to 0$: **Infinite penalty!**

### Reverse KL: $D_{KL}(Q \| P)$

**"Mode-seeking" or "Exclusive"**

- Expectation is over $Q$ (our approximation)
- Penalizes $Q$ for having mass where $P$ is low
- **Allows $Q$ to ignore modes of $P$**
- Results in underdispersed, mode-focused approximations

$$D_{KL}(Q \| P) = \mathbb{E}_{x \sim Q}\left[\log \frac{Q(x)}{P(x)}\right]$$

When $Q(x) > 0$ but $P(x) \to 0$: **Infinite penalty!**

### Visual Intuition

Imagine fitting a unimodal Gaussian to a bimodal distribution:

| KL Direction | Behavior | Result |
|--------------|----------|--------|
| Forward $D_{KL}(P \| Q)$ | Must cover both modes | Gaussian spreads wide, covers everything |
| Reverse $D_{KL}(Q \| P)$ | Can ignore one mode | Gaussian locks onto one mode, ignores other |

---

## 4. KL Divergence in Machine Learning

### Maximum Likelihood = Minimizing Forward KL

When we minimize negative log-likelihood:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log Q_\theta(x_i)$$

We're actually minimizing:
$$D_{KL}(\hat{P}_{data} \| Q_\theta)$$

where $\hat{P}_{data}$ is the empirical data distribution.

**This is why ML models try to cover all the data!**

### Variational Inference = Minimizing Reverse KL

In variational inference, we minimize:
$$D_{KL}(Q_\phi(z|x) \| P(z|x))$$

This is reverse KL—our approximation $Q$ can ignore modes of the true posterior.

**This is why VI tends to underestimate uncertainty!**

---

## 5. Closed-Form KL for Common Distributions

### Two Bernoulli Distributions

$P = \text{Bernoulli}(p)$, $Q = \text{Bernoulli}(q)$

$$D_{KL}(P \| Q) = p \log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$$

### Two Categorical Distributions

$P = \text{Cat}(p_1, \ldots, p_K)$, $Q = \text{Cat}(q_1, \ldots, q_K)$

$$D_{KL}(P \| Q) = \sum_{k=1}^K p_k \log\frac{p_k}{q_k}$$

### Two Univariate Gaussians

$P = \mathcal{N}(\mu_1, \sigma_1^2)$, $Q = \mathcal{N}(\mu_2, \sigma_2^2)$

$$D_{KL}(P \| Q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### Two Multivariate Gaussians

$P = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$, $Q = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$

$$D_{KL}(P \| Q) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} - d + \text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]$$

where $d$ is the dimensionality.

### Standard Normal Prior (VAE Special Case)

For $Q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $P = \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$D_{KL}(Q \| P) = \frac{1}{2}\sum_{j=1}^d \left[\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right]$$

This is the VAE latent regularization term!

---

## 6. Information-Theoretic Interpretation

### Relative Entropy

KL divergence is also called **relative entropy**. It measures the "information gained" when we update from prior $Q$ to posterior $P$:

$$D_{KL}(P \| Q) = \text{Information in } P \text{ relative to } Q$$

### Bayesian Update Perspective

If $Q$ is your prior and $P$ is your posterior:
- $D_{KL}(P \| Q)$ = information gained from data
- Large KL = data was surprising given prior
- Small KL = data was expected given prior

---

## 7. Why KL Divergence Matters for VAEs

In VAEs, we minimize:
$$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{KL}(q(z|x) \| p(z))$$

The KL term $D_{KL}(q(z|x) \| p(z))$:

1. **Regularizes** the latent space
2. **Compresses** information—forces encoder to discard noise
3. **Enables sampling**—keeps $q(z|x)$ close to prior $p(z)$
4. **Rate-distortion tradeoff**—balances reconstruction vs compression

### The Information Bottleneck View

- **Without KL:** Encoder could memorize each input (no compression)
- **With KL:** Encoder must use a "budget" of information
- Larger KL weight → more compression → more regularization

---

## 8. Summary

| Concept | Formula | Intuition |
|---------|---------|-----------|
| KL (discrete) | $\sum_x P(x) \log \frac{P(x)}{Q(x)}$ | Extra bits from wrong code |
| KL (continuous) | $\int p(x) \log \frac{p(x)}{q(x)} dx$ | Same, for densities |
| Forward KL | $D_{KL}(P \| Q)$ | Mean-seeking, covers all modes |
| Reverse KL | $D_{KL}(Q \| P)$ | Mode-seeking, can ignore modes |
| ML training | Minimizes $D_{KL}(\hat{P} \| Q_\theta)$ | Cover the data |
| VI | Minimizes $D_{KL}(Q \| P)$ | Approximate posterior |

### Key Takeaways

1. **KL divergence is asymmetric**—the direction matters!
2. **Forward KL** (ML) tries to cover everything
3. **Reverse KL** (VI) can be selective
4. **Closed forms exist** for Gaussians—crucial for VAEs
5. **KL to prior** in VAEs creates an information bottleneck

---

## Next: Normal Distributions

To fully understand the KL terms in VAEs, we need to deeply understand Gaussian distributions and their information-theoretic properties.
