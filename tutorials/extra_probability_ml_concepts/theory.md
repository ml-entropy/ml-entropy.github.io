# Tutorial 06: Probability Concepts in Machine Learning

## Introduction

In ML, we constantly see notation like $P(y|x)$, "likelihood", "posterior", "prior". These terms have precise meanings that are often confused. This tutorial will give you a **rock-solid understanding** with careful derivations.

---

## Part 1: Basic Probability Notation

### 1.1 What Does P(x) Mean?

$P(x)$ or $P(X = x)$ is the **probability** that random variable $X$ takes value $x$.

**Example**: Rolling a fair die
- $P(X = 3) = \frac{1}{6}$ — probability of rolling a 3
- $P(X = 7) = 0$ — impossible outcome

### 1.2 The Cast of Characters

| Notation | Name | Meaning |
|----------|------|---------|
| $P(x)$ | **Marginal probability** | Probability of $x$ alone, ignoring everything else |
| $P(y)$ | **Marginal probability** | Probability of $y$ alone |
| $P(x, y)$ | **Joint probability** | Probability of $x$ AND $y$ together |
| $P(x \mid y)$ | **Conditional probability** | Probability of $x$ GIVEN that $y$ happened |
| $P(y \mid x)$ | **Conditional probability** | Probability of $y$ GIVEN that $x$ happened |

---

## Part 2: Joint, Marginal, and Conditional Probability

### 2.1 Joint Probability: P(x, y)

**Joint probability** is the probability that BOTH events happen.

**Example**: Drawing two cards (without replacement)
- $P(\text{first card Ace AND second card King})$
- This is a joint probability

**Key property**: $P(x, y) = P(y, x)$ (order doesn't matter in the notation)

### 2.2 Marginal Probability: P(x)

**Marginal probability** is the probability of one variable, **summing out** (marginalizing over) all other variables.

**Derivation** (from joint):
$$P(x) = \sum_y P(x, y)$$

**Why "marginal"?** 
Historically, when you had a joint probability table, you'd write row/column sums in the margins:

|         | $y=0$ | $y=1$ | **Margin** |
|---------|-------|-------|------------|
| $x=0$   | 0.2   | 0.1   | **0.3** = P(x=0) |
| $x=1$   | 0.3   | 0.4   | **0.7** = P(x=1) |
| **Margin** | **0.5** | **0.5** | 1.0 |

The marginal $P(x=0) = P(x=0, y=0) + P(x=0, y=1) = 0.2 + 0.1 = 0.3$

### 2.3 Conditional Probability: P(x|y)

**Conditional probability** is the probability of $x$ **given that we know** $y$ happened.

**Intuition**: We're "restricting our universe" to only cases where $y$ occurred.

**Definition** (this is the starting point):
$$P(x \mid y) = \frac{P(x, y)}{P(y)}$$

**Derivation of why this makes sense**:

Consider 1000 people:
- 300 have disease ($D$), 700 don't
- 250 test positive ($T$), 750 test negative
- Joint breakdown:

| | $T=+$ | $T=-$ | Total |
|---|-------|-------|-------|
| $D=yes$ | 240 | 60 | 300 |
| $D=no$ | 10 | 690 | 700 |
| Total | 250 | 750 | 1000 |

**Question**: $P(D=yes \mid T=+)$ = probability of disease given positive test?

**Answer**: Out of 250 positive tests, 240 have disease:
$$P(D=yes \mid T=+) = \frac{240}{250} = 0.96$$

**But notice**:
- $P(D=yes, T=+) = \frac{240}{1000} = 0.24$ (joint)
- $P(T=+) = \frac{250}{1000} = 0.25$ (marginal)
- $\frac{P(D=yes, T=+)}{P(T=+)} = \frac{0.24}{0.25} = 0.96$ ✓

This confirms: $P(x \mid y) = \frac{P(x,y)}{P(y)}$

---

## Part 3: The Product Rule (Chain Rule)

### 3.1 Derivation

Starting from the definition of conditional probability:
$$P(x \mid y) = \frac{P(x, y)}{P(y)}$$

**Rearranging**:
$$P(x, y) = P(x \mid y) \cdot P(y)$$

This is the **product rule** or **chain rule** of probability.

### 3.2 Symmetry Gives Us Two Forms

Since $P(x, y) = P(y, x)$:

$$P(x, y) = P(x \mid y) \cdot P(y) = P(y \mid x) \cdot P(x)$$

Both are valid decompositions of the same joint probability!

### 3.3 Extension to Multiple Variables

$$P(a, b, c) = P(a \mid b, c) \cdot P(b \mid c) \cdot P(c)$$

Or any other ordering:
$$P(a, b, c) = P(c \mid a, b) \cdot P(b \mid a) \cdot P(a)$$

---

## Part 4: Law of Total Probability

### 4.1 The Formula

If events $B_1, B_2, ..., B_n$ are **mutually exclusive** and **exhaustive** (they partition the sample space):

$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)$$

### 4.2 Derivation

**Step 1**: $A$ can be decomposed based on which $B_i$ occurred:
$$A = (A \cap B_1) \cup (A \cap B_2) \cup ... \cup (A \cap B_n)$$

**Step 2**: Since $B_i$ are mutually exclusive, so are $(A \cap B_i)$:
$$P(A) = \sum_{i=1}^{n} P(A \cap B_i)$$

**Step 3**: Apply product rule to each term:
$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)$$

### 4.3 Intuition

Think of $P(A)$ as a **weighted average** of conditional probabilities:
- $P(A \mid B_i)$ = probability of $A$ in "world $B_i$"
- $P(B_i)$ = how likely is "world $B_i$"
- Sum over all possible worlds = overall probability

### 4.4 Example: Disease Diagnosis

- Disease prevalence: $P(D) = 0.01$ (1% have disease)
- Test sensitivity: $P(T+ \mid D) = 0.99$ (99% true positive rate)
- Test specificity: $P(T- \mid \neg D) = 0.95$ (5% false positive rate)

**Question**: What's the probability of testing positive, $P(T+)$?

**Using Law of Total Probability**:
$$P(T+) = P(T+ \mid D) \cdot P(D) + P(T+ \mid \neg D) \cdot P(\neg D)$$
$$P(T+) = 0.99 \times 0.01 + 0.05 \times 0.99$$
$$P(T+) = 0.0099 + 0.0495 = 0.0594$$

About 6% test positive overall.

---

## Part 5: Bayes' Theorem

### 5.1 The Formula

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

### 5.2 Derivation (Three Methods)

#### Method 1: From Definition

Start with definition of conditional probability:
$$P(A \mid B) = \frac{P(A, B)}{P(B)}$$

Use product rule on numerator:
$$P(A, B) = P(B \mid A) \cdot P(A)$$

Substitute:
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

#### Method 2: From Symmetry of Joint

We have two ways to factor $P(A, B)$:
$$P(A, B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Therefore:
$$P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Divide both sides by $P(B)$:
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

#### Method 3: Verbal Derivation

"The probability of A given B" 
= "How often A and B occur together, relative to how often B occurs"
= $\frac{P(A,B)}{P(B)}$
= $\frac{P(B|A) \cdot P(A)}{P(B)}$

### 5.3 Extended Form (with Total Probability)

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid \neg A) \cdot P(\neg A)}$$

This is often more useful because the denominator terms are often known.

### 5.4 Example: Medical Diagnosis (continued)

Given our earlier setup:
- $P(D) = 0.01$
- $P(T+ \mid D) = 0.99$
- $P(T+ \mid \neg D) = 0.05$

**Question**: If someone tests positive, what's the probability they have the disease?

$$P(D \mid T+) = \frac{P(T+ \mid D) \cdot P(D)}{P(T+)}$$

We calculated $P(T+) = 0.0594$ earlier.

$$P(D \mid T+) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594} \approx 0.167$$

**Only 17% chance of disease even with positive test!**

This is the "base rate fallacy" — low prevalence means many false positives.

---

## Part 6: Probability vs Likelihood

This is one of the most confusing distinctions in ML. Let's clear it up.

### 6.1 Setup

We have:
- Data: $x$ (observed)
- Parameters: $\theta$ (unknown, we want to estimate)
- A probabilistic model: $P(x \mid \theta)$

### 6.2 Probability (Function of Data)

**Probability** asks: "Given fixed parameters $\theta$, how likely is data $x$?"

$$P(x \mid \theta) \text{ as a function of } x$$

- $\theta$ is **fixed** (known)
- $x$ varies (what might we observe?)
- Sums/integrates to 1 over $x$

**Example**: Coin with $\theta = 0.7$ (probability of heads)

| $x$ (# heads in 10 flips) | $P(x \mid \theta=0.7)$ |
|---------------------------|------------------------|
| 0 | 0.0000059 |
| 5 | 0.103 |
| 7 | 0.267 |
| 10 | 0.028 |
| **Sum** | **1.0** |

### 6.3 Likelihood (Function of Parameters)

**Likelihood** asks: "Given observed data $x$, how likely is parameter $\theta$?"

$$\mathcal{L}(\theta \mid x) = P(x \mid \theta) \text{ as a function of } \theta$$

- $x$ is **fixed** (we observed it)
- $\theta$ varies (which parameter explains our data best?)
- Does **NOT** sum/integrate to 1 over $\theta$!

**Same formula, different perspective!**

**Example**: We observed 7 heads in 10 flips. Which $\theta$ best explains this?

| $\theta$ | $\mathcal{L}(\theta \mid x=7) = P(x=7 \mid \theta)$ |
|----------|-----------------------------------------------------|
| 0.3 | 0.009 |
| 0.5 | 0.117 |
| 0.7 | 0.267 |
| 0.9 | 0.057 |
| **Sum** | **0.45** (not 1!) |

Maximum likelihood at $\theta = 0.7$ — makes sense!

### 6.4 The Key Distinction

| | Probability | Likelihood |
|--|-------------|------------|
| **What varies?** | Data $x$ | Parameters $\theta$ |
| **What's fixed?** | Parameters $\theta$ | Data $x$ |
| **Sums to 1?** | Yes (over $x$) | No |
| **Question** | "What data might I see?" | "What parameters explain my data?" |

### 6.5 Why the Distinction Matters

In ML, we **observe data** and want to **find parameters**. So we use **likelihood**.

Maximum Likelihood Estimation (MLE):
$$\hat{\theta}_{MLE} = \arg\max_\theta \mathcal{L}(\theta \mid x) = \arg\max_\theta P(x \mid \theta)$$

We're NOT computing probability of $\theta$ — we're computing likelihood.

---

## Part 7: Prior, Posterior, and Bayesian Inference

### 7.1 The Bayesian Framework

Bayesians treat parameters $\theta$ as **random variables** with their own distributions.

**Bayes' Theorem for Parameters**:
$$P(\theta \mid x) = \frac{P(x \mid \theta) \cdot P(\theta)}{P(x)}$$

### 7.2 The Terminology

| Term | Symbol | Meaning |
|------|--------|---------|
| **Prior** | $P(\theta)$ | Our belief about $\theta$ BEFORE seeing data |
| **Likelihood** | $P(x \mid \theta)$ | How likely the data is, given parameters |
| **Evidence** | $P(x)$ | Total probability of observed data |
| **Posterior** | $P(\theta \mid x)$ | Our belief about $\theta$ AFTER seeing data |

### 7.3 Intuition for Each Term

#### Prior: $P(\theta)$

Your **initial belief** before seeing any data.

**Examples**:
- "I think the coin is probably fair" → $P(\theta) = \text{Beta}(\alpha=10, \beta=10)$, peaked at 0.5
- "I have no idea" → $P(\theta) = \text{Uniform}(0, 1)$
- "The coin is probably biased toward heads" → $P(\theta) = \text{Beta}(\alpha=5, \beta=2)$

#### Likelihood: $P(x \mid \theta)$

How well parameters $\theta$ **explain** the observed data $x$.

**Example**: 7 heads in 10 flips
- $\theta = 0.5$: $P(x \mid \theta) = \binom{10}{7}(0.5)^{10} = 0.117$
- $\theta = 0.7$: $P(x \mid \theta) = \binom{10}{7}(0.7)^7(0.3)^3 = 0.267$

#### Evidence: $P(x)$

The **normalizing constant** — ensures posterior sums/integrates to 1.

$$P(x) = \int P(x \mid \theta) P(\theta) d\theta$$

Often intractable to compute! (This is why we need variational inference, MCMC, etc.)

#### Posterior: $P(\theta \mid x)$

Your **updated belief** after seeing data. Combines prior knowledge with evidence from data.

### 7.4 The Bayesian Update Process

```
    Prior           Likelihood         Posterior
    P(θ)      ×     P(x|θ)       ∝     P(θ|x)
  
  [Initial    ]   [Evidence from]   [Updated    ]
  [belief     ] × [data         ] = [belief     ]
```

**Key insight**: Posterior $\propto$ Prior $\times$ Likelihood

(The evidence $P(x)$ just normalizes)

### 7.5 Example: Coin Flipping

**Setup**:
- Prior: $P(\theta) = \text{Uniform}(0, 1)$ (no prior knowledge)
- Data: 7 heads in 10 flips
- Likelihood: $P(x \mid \theta) = \binom{10}{7} \theta^7 (1-\theta)^3$

**Posterior**:
$$P(\theta \mid x) \propto \theta^7 (1-\theta)^3 \times 1$$
$$P(\theta \mid x) = \text{Beta}(8, 4)$$

The posterior is peaked at $\theta = 0.7$, but with uncertainty.

### 7.6 MLE vs MAP vs Full Bayesian

| Method | What it computes | Formula |
|--------|-----------------|---------|
| **MLE** | Best single $\theta$ (ignores prior) | $\arg\max_\theta P(x \mid \theta)$ |
| **MAP** | Best single $\theta$ (with prior) | $\arg\max_\theta P(\theta \mid x)$ |
| **Full Bayesian** | Entire posterior distribution | $P(\theta \mid x)$ |

---

## Part 8: Application to Machine Learning

### 8.1 Supervised Learning Notation

In classification/regression:
- $x$ = input features
- $y$ = output label/value
- $\theta$ = model parameters (weights)

**The probabilistic model**: $P(y \mid x, \theta)$

### 8.2 Training = Maximizing Likelihood

Given dataset $\mathcal{D} = \{(x_1, y_1), ..., (x_n, y_n)\}$:

$$\mathcal{L}(\theta) = \prod_{i=1}^{n} P(y_i \mid x_i, \theta)$$

**MLE**: $\hat{\theta} = \arg\max_\theta \mathcal{L}(\theta)$

**Log-likelihood** (more practical):
$$\log \mathcal{L}(\theta) = \sum_{i=1}^{n} \log P(y_i \mid x_i, \theta)$$

### 8.3 Loss Functions Are Negative Log-Likelihoods!

| Assumption about $P(y \mid x, \theta)$ | Negative Log-Likelihood | Loss Function Name |
|----------------------------------------|------------------------|-------------------|
| $y \mid x \sim \mathcal{N}(\mu(x; \theta), \sigma^2)$ | $\frac{1}{2\sigma^2}\|y - \mu\|^2 + \text{const}$ | **MSE Loss** |
| $y \mid x \sim \text{Categorical}(f(x; \theta))$ | $-\sum y_i \log \hat{y}_i$ | **Cross-Entropy Loss** |
| $y \mid x \sim \text{Bernoulli}(\sigma(f(x; \theta)))$ | $-y\log\hat{y} - (1-y)\log(1-\hat{y})$ | **Binary Cross-Entropy** |

**Minimizing loss = Maximizing likelihood!**

### 8.4 Regularization = Adding a Prior

**MAP estimation**:
$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta \mid \mathcal{D}) = \arg\max_\theta P(\mathcal{D} \mid \theta) P(\theta)$$

Taking log:
$$\hat{\theta}_{MAP} = \arg\max_\theta \left[ \log P(\mathcal{D} \mid \theta) + \log P(\theta) \right]$$

| Prior $P(\theta)$ | $\log P(\theta)$ | Regularization |
|-------------------|------------------|----------------|
| $\mathcal{N}(0, \sigma^2)$ | $-\frac{\|\theta\|^2}{2\sigma^2}$ | **L2 (Ridge)** |
| $\text{Laplace}(0, b)$ | $-\frac{\|\theta\|_1}{b}$ | **L1 (Lasso)** |

**L2 regularization = Gaussian prior on weights!**

### 8.5 Generative vs Discriminative Models

**Discriminative** (most neural networks):
- Model: $P(y \mid x)$ directly
- "Given input, predict output"
- Example: Logistic regression, most DNNs

**Generative**:
- Model: $P(x, y) = P(x \mid y) P(y)$
- Can generate new samples
- Use Bayes to get $P(y \mid x) = \frac{P(x \mid y) P(y)}{P(x)}$
- Example: Naive Bayes, VAEs, GANs

---

## Part 9: Summary — The Complete Picture

### The Probability Hierarchy

```
P(x)        Marginal: x alone
P(y)        Marginal: y alone  
P(x,y)      Joint: x AND y together
P(x|y)      Conditional: x given y happened
P(y|x)      Conditional: y given x happened
```

### The Key Formulas

| Name | Formula |
|------|---------|
| **Product Rule** | $P(x,y) = P(x \mid y) P(y) = P(y \mid x) P(x)$ |
| **Marginalization** | $P(x) = \sum_y P(x, y)$ |
| **Total Probability** | $P(A) = \sum_i P(A \mid B_i) P(B_i)$ |
| **Bayes' Theorem** | $P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$ |

### The Bayesian ML Picture

$$\underbrace{P(\theta \mid x)}_{\text{Posterior}} = \frac{\overbrace{P(x \mid \theta)}^{\text{Likelihood}} \cdot \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(x)}_{\text{Evidence}}}$$

- **Prior**: What we believe before data
- **Likelihood**: How well $\theta$ explains data
- **Posterior**: What we believe after data
- **Evidence**: Normalizing constant (often intractable)

---

## Exercises

1. **Joint Table**: Given joint probabilities P(X,Y), compute all marginals and conditionals.

2. **Bayes Practice**: A factory has two machines. Machine A produces 60% of items, Machine B produces 40%. Machine A has 5% defect rate, Machine B has 2%. An item is defective. What's the probability it came from Machine A?

3. **Likelihood vs Probability**: You flip a coin 20 times and get 15 heads. 
   - Write the likelihood function $\mathcal{L}(\theta)$
   - Find the MLE
   - If your prior is Beta(2,2), find the MAP estimate

4. **Loss Derivation**: Show that minimizing MSE loss is equivalent to MLE with Gaussian noise assumption.

5. **Regularization as Prior**: Show that L2 regularization with $\lambda$ corresponds to a Gaussian prior with what variance?
