# Tutorial 00: Probability Foundations

## 🎯 Why This Tutorial?

Before diving into entropy, KL divergence, and VAEs, we need rock-solid intuition about **probability distributions**. Many ML concepts become confusing because the foundations are shaky.

**This tutorial answers:**
- What is the difference between probability and probability distribution?
- What is a random variable, really?
- What are PMF and PDF, and why do we need both?
- How do we go from discrete to continuous?
- What do distributions "look like" and how do we work with them?

---

## 1. Events, Outcomes, and Probability

### The Sample Space

When we have randomness, we first define all possible **outcomes**. This set is called the **sample space** $\Omega$.

**Examples:**

| Experiment | Sample Space $\Omega$ |
|------------|----------------------|
| Coin flip | $\{H, T\}$ |
| Die roll | $\{1, 2, 3, 4, 5, 6\}$ |
| Card draw | 52 possible cards |
| Temperature | $\mathbb{R}$ (any real number) |
| Image pixel | $[0, 255]$ or $[0, 1]$ |

### Events

An **event** is a subset of the sample space—a collection of outcomes we care about.

- Event "coin lands heads": $\{H\}$
- Event "die shows even": $\{2, 4, 6\}$
- Event "temperature below freezing": $(-\infty, 0]$

### Probability of an Event

**Probability** $P(A)$ assigns a number in $[0, 1]$ to an event $A$.

**Axioms (Kolmogorov):**
1. $P(A) \geq 0$ for all events $A$
2. $P(\Omega) = 1$ (something must happen)
3. $P(A \cup B) = P(A) + P(B)$ if $A$ and $B$ are disjoint

### 🔑 Key Insight: Probability vs. Probability Distribution

| Concept | What it is | Example |
|---------|-----------|---------|
| **Probability** | A single number for one event | $P(\text{heads}) = 0.5$ |
| **Probability Distribution** | A complete description of ALL probabilities | The full assignment of probabilities to every outcome |

**A probability distribution tells you everything; a probability is just one question answered.**

---

## 2. Random Variables: The Bridge

### The Problem

Sample spaces can be abstract (colors, categories, cards). We want to do math with numbers.

### The Solution: Random Variables

A **random variable** $X$ is a function that maps outcomes to numbers:
$$X: \Omega \to \mathbb{R}$$

**Examples:**
- Die roll: $X(\omega) = \omega$ (already a number)
- Coin flip: $X(H) = 1$, $X(T) = 0$
- Card draw: $X(\text{card}) = \text{value of card}$

### Why "Variable"?

It's called a "variable" because before the experiment, we don't know which value it will take. After the experiment, it becomes a concrete number.

### Types of Random Variables

| Type | Values | Example |
|------|--------|---------|
| **Discrete** | Countable set | Die roll: $\{1,2,3,4,5,6\}$ |
| **Continuous** | Uncountable (intervals) | Height: any value in $[0, 300]$ cm |

This distinction is **crucial**—it determines whether we use PMF or PDF.

---

## 3. Discrete Distributions: PMF

### The Probability Mass Function (PMF)

For a **discrete** random variable $X$, we can ask: "What's the probability that $X$ equals exactly $x$?"

The **PMF** $p(x)$ or $P(X = x)$ gives this probability for each possible value:

$$p(x) = P(X = x)$$

### Properties of PMF

1. **Non-negative:** $p(x) \geq 0$ for all $x$
2. **Sums to 1:** $\sum_x p(x) = 1$

### Example: Fair Die

$$p(x) = \begin{cases} \frac{1}{6} & \text{if } x \in \{1,2,3,4,5,6\} \\ 0 & \text{otherwise} \end{cases}$$

Verification: $\frac{1}{6} + \frac{1}{6} + \frac{1}{6} + \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = 1$ ✓

### Example: Biased Coin

Let $X = 1$ for heads, $X = 0$ for tails, with $P(\text{heads}) = 0.7$:

$$p(x) = \begin{cases} 0.7 & \text{if } x = 1 \\ 0.3 & \text{if } x = 0 \\ 0 & \text{otherwise} \end{cases}$$

### Computing Probabilities from PMF

**Probability of an event = sum of PMF over outcomes in the event**

Example: "Die shows at most 3"
$$P(X \leq 3) = p(1) + p(2) + p(3) = \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = \frac{1}{2}$$

---

## 4. The Continuous Paradox

### The Problem

What if $X$ can take any real value? For example, height can be 170.000000... cm.

**Attempt:** Let's compute $P(X = 170.5)$ for a continuous variable.

There are infinitely many possible values, and probability must sum to 1. If we assign any positive probability to each value, the sum would be infinite!

**Conclusion:** For continuous random variables:
$$P(X = x) = 0 \quad \text{for any specific } x$$

### Wait, Everything Has Zero Probability?

Yes! But that doesn't mean the distribution is useless. We shift focus from **exact values** to **intervals**.

---

## 5. Continuous Distributions: PDF

### The Probability Density Function (PDF)

For a **continuous** random variable, we define the **PDF** $f(x)$ such that:

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

**The PDF is NOT a probability!** It's a **density**—probability per unit length.

### Properties of PDF

1. **Non-negative:** $f(x) \geq 0$ for all $x$
2. **Integrates to 1:** $\int_{-\infty}^{\infty} f(x) \, dx = 1$

### 🔑 Key Intuition: PDF as a Limit

Imagine we discretize continuous $X$ into bins of width $\Delta x$. The probability of being in a small bin around $x$ is:

$$P(x \leq X \leq x + \Delta x) \approx f(x) \cdot \Delta x$$

As $\Delta x \to 0$:
- The probability of any exact value → 0
- But the **density** $f(x)$ remains meaningful

### PDF Can Be Greater Than 1!

Since PDF is a density, not a probability, $f(x) > 1$ is perfectly valid.

**Example:** Uniform distribution on $[0, 0.5]$:
$$f(x) = \begin{cases} 2 & \text{if } 0 \leq x \leq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Check: $\int_0^{0.5} 2 \, dx = 2 \times 0.5 = 1$ ✓

---

## 6. PMF vs PDF: A Comparison

### The Analogy: Mass vs Density

| Concept | Discrete (PMF) | Continuous (PDF) |
|---------|---------------|------------------|
| Physical analogy | Point masses | Continuous fluid |
| $p(x)$ or $f(x)$ | Probability | Density |
| Can be > 1? | No | Yes |
| $P(X = x)$ | $p(x)$ | 0 |
| $P(a \leq X \leq b)$ | $\sum_{a \leq x \leq b} p(x)$ | $\int_a^b f(x) dx$ |
| Normalization | $\sum_x p(x) = 1$ | $\int f(x) dx = 1$ |

### Unifying View: Measure Theory

In advanced probability, both are unified using **measures**:
- Discrete: counting measure (sum)
- Continuous: Lebesgue measure (integral)

But for practical purposes, just remember:
- **Discrete → Sum**
- **Continuous → Integrate**

---

## 7. The Cumulative Distribution Function (CDF)

### Definition

The **CDF** $F(x)$ works for both discrete and continuous:

$$F(x) = P(X \leq x)$$

### For Discrete

$$F(x) = \sum_{t \leq x} p(t)$$

### For Continuous

$$F(x) = \int_{-\infty}^{x} f(t) \, dt$$

And conversely:
$$f(x) = \frac{dF(x)}{dx}$$

### Properties of CDF

1. $F(-\infty) = 0$
2. $F(+\infty) = 1$
3. $F$ is non-decreasing
4. $P(a < X \leq b) = F(b) - F(a)$

### Why CDF is Useful

- Computing probabilities of intervals
- Comparing distributions
- Generating random samples (inverse transform method)

---

## 8. Key Distribution Characteristics

### Expected Value (Mean)

**Discrete:**
$$\mathbb{E}[X] = \sum_x x \cdot p(x)$$

**Continuous:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

**Intuition:** The "center of mass" of the distribution.

### Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Discrete:**
$$\text{Var}(X) = \sum_x (x - \mu)^2 \cdot p(x)$$

**Continuous:**
$$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot f(x) \, dx$$

**Intuition:** Average squared distance from the mean—measures "spread."

### Standard Deviation

$$\sigma = \sqrt{\text{Var}(X)}$$

In the same units as $X$ (variance is in squared units).

---

## 9. Common Distributions

### Discrete Distributions

| Distribution | PMF | Mean | Variance | Use Case |
|-------------|-----|------|----------|----------|
| Bernoulli$(p)$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | Single yes/no trial |
| Binomial$(n,p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Count of successes |
| Categorical | $(p_1, ..., p_K)$ | — | — | One of K classes |
| Poisson$(\lambda)$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Count of rare events |

### Continuous Distributions

| Distribution | PDF | Mean | Variance | Use Case |
|-------------|-----|------|----------|----------|
| Uniform$(a,b)$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Equal likelihood |
| Normal$(\mu, \sigma^2)$ | $\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | Natural variation |
| Exponential$(\lambda)$ | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | Waiting times |

---

## 10. From Discrete to Continuous: The Limiting Process

### Motivation

Understanding how discrete becomes continuous helps in many derivations (entropy, KL, etc.).

### The Recipe

1. **Start discrete:** Divide the range into bins of width $\Delta x$
2. **Compute discrete quantities** using sums and PMF
3. **Take limit:** Let $\Delta x \to 0$
4. **Sums become integrals:** $\sum \to \int$
5. **PMF becomes PDF:** $p(x_i) \to f(x) \Delta x$

### Example: Mean

**Discrete approximation:**
$$\mathbb{E}[X] \approx \sum_i x_i \cdot p(x_i)$$

**With bins:** $p(x_i) \approx f(x_i) \cdot \Delta x$
$$\mathbb{E}[X] \approx \sum_i x_i \cdot f(x_i) \cdot \Delta x$$

**Limit as $\Delta x \to 0$:**
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

This pattern works for **everything**:
- Variance
- Entropy
- KL divergence
- Expectations of any function

---

## 11. Joint and Conditional Distributions

### Joint Distribution

For two random variables $X$ and $Y$:

**Discrete:**
$$p(x, y) = P(X = x \text{ and } Y = y)$$

**Continuous:**
$$f(x, y) \text{ such that } P((X,Y) \in A) = \iint_A f(x,y) \, dx \, dy$$

### Marginal Distribution

"Forget" about one variable:

**Discrete:**
$$p(x) = \sum_y p(x, y)$$

**Continuous:**
$$f_X(x) = \int_{-\infty}^{\infty} f(x, y) \, dy$$

### Conditional Distribution

Probability of $X$ given we know $Y = y$:

$$p(x | y) = \frac{p(x, y)}{p(y)}$$

$$f(x | y) = \frac{f(x, y)}{f_Y(y)}$$

### Independence

$X$ and $Y$ are independent if:
$$p(x, y) = p(x) \cdot p(y) \quad \text{or} \quad f(x,y) = f_X(x) \cdot f_Y(y)$$

Equivalently: $p(x|y) = p(x)$—knowing $Y$ tells us nothing about $X$.

---

## 12. Why This Matters for ML

### Distributions Are Models

In ML, we model data as coming from some distribution:
- Classification: $P(y | x)$ — categorical distribution over classes
- Regression: $p(y | x) = \mathcal{N}(y | f(x), \sigma^2)$ — Gaussian with learned mean
- Generation (VAE): $p(x | z)$ — distribution over data given latent code

### Loss Functions Come from Distributions

| Loss | Assumed Distribution | Connection |
|------|---------------------|------------|
| MSE | Gaussian | $-\log \mathcal{N}(y \| \hat{y}, \sigma^2) \propto (y - \hat{y})^2$ |
| Cross-entropy | Categorical | $-\log p(y \| \hat{y})$ |
| MAE | Laplace | $-\log \text{Laplace}(y \| \hat{y}, b) \propto |y - \hat{y}|$ |

### Sampling and Generation

To generate data, we need to **sample from distributions**:
- VAE: Sample $z \sim \mathcal{N}(0, I)$, then $x \sim p(x|z)$
- Diffusion: Iteratively sample to denoise

---

## Summary

### Key Concepts

| Concept | Discrete | Continuous |
|---------|----------|------------|
| Values | Countable | Uncountable |
| Distribution | PMF $p(x)$ | PDF $f(x)$ |
| $P(X = x)$ | $p(x)$ | 0 |
| $P(a \leq X \leq b)$ | $\sum_{a \leq x \leq b} p(x)$ | $\int_a^b f(x) dx$ |
| Normalization | $\sum p(x) = 1$ | $\int f(x) dx = 1$ |
| Can exceed 1? | No | Yes (it's density) |

### The Pattern

**Discrete → Continuous:**
1. Replace sums with integrals
2. Replace PMF with PDF × dx
3. Keep the structure, change the calculus

### Next Steps

With this foundation, we can now understand:
- **Entropy:** Expected information (sum/integral of $-p \log p$)
- **KL Divergence:** Extra bits from wrong distribution
- **Why Gaussians:** Maximum entropy given mean and variance
- **VAEs:** Distributions over latent spaces

---

## Next: Entropy Fundamentals

Now that we understand what distributions are, we're ready to ask: **How do we measure the "information" or "uncertainty" in a distribution?**
