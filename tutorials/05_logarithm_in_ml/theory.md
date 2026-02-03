# Tutorial 05: The Logarithm in Machine Learning — Convenience or Fundamental?

## The Central Question

> "If we had infinitely precise computers that could handle arbitrarily small numbers and compute derivatives of products perfectly, would we still need logarithms in ML?"

This is a profound question that gets to the heart of why neural networks work. Let's explore it deeply.

**Spoiler**: Logarithm is **not** just a computational trick. It's mathematically fundamental to how learning works.

---

## Part 1: What is a Logarithm? (Quick Review)

### Definition

The logarithm answers: "What power do I raise the base to, to get this number?"

$$\log_b(x) = y \quad \Leftrightarrow \quad b^y = x$$

### Key Properties

| Property | Formula | Why it matters |
|----------|---------|----------------|
| Product → Sum | $\log(xy) = \log(x) + \log(y)$ | Independent probabilities become additive |
| Power → Multiply | $\log(x^n) = n \log(x)$ | Repeated events scale linearly |
| Quotient → Difference | $\log(x/y) = \log(x) - \log(y)$ | Ratios become differences |
| $\log(1) = 0$ | | Certainty has zero information |
| Monotonic | $x > y \Rightarrow \log(x) > \log(y)$ | Preserves ordering (same optima) |

### The Uniqueness Theorem

**Theorem**: The logarithm is the **ONLY** continuous function $f$ satisfying:
$$f(xy) = f(x) + f(y)$$

*Proof sketch*:
1. Let $f(xy) = f(x) + f(y)$ for all $x, y > 0$
2. Setting $x = y = 1$: $f(1) = 2f(1) \Rightarrow f(1) = 0$
3. Setting $y = x$: $f(x^2) = 2f(x)$
4. By induction: $f(x^n) = nf(x)$ for integers
5. For rationals: $f(x^{p/q}) = \frac{p}{q}f(x)$
6. By continuity, extending to reals: $f(x) = f(e) \cdot \ln(x)$

**This is why log appears everywhere** — it's the unique bridge between multiplication and addition.

---

## Part 2: Why ML Uses Logarithms (The Usual Explanations)

### 2.1 Numerical Stability

Probabilities of sequences multiply:
$$P(\text{sequence}) = P(x_1) \cdot P(x_2) \cdot ... \cdot P(x_n)$$

With $n = 1000$ words and average $P(x_i) = 0.01$:
$$P = (0.01)^{1000} = 10^{-2000}$$

This underflows to 0 in any floating-point system (float64 bottoms out around $10^{-308}$).

With log:
$$\log P = 1000 \cdot \log(0.01) = -2000$$

Perfectly representable!

### 2.2 Simpler Gradients

**Without log** (direct MLE):
$$L = \prod_{i=1}^{n} P(x_i | \theta)$$

$$\frac{\partial L}{\partial \theta} = \sum_{i=1}^{n} \left( \frac{\partial P(x_i|\theta)}{\partial \theta} \cdot \prod_{j \neq i} P(x_j|\theta) \right)$$

This is a sum of products — messy!

**With log**:
$$\ell = \log L = \sum_{i=1}^{n} \log P(x_i | \theta)$$

$$\frac{\partial \ell}{\partial \theta} = \sum_{i=1}^{n} \frac{1}{P(x_i|\theta)} \cdot \frac{\partial P(x_i|\theta)}{\partial \theta}$$

Clean sum of local terms!

### 2.3 These Seem Like "Just" Computational Conveniences...

So the question remains: **with perfect computers, could we skip the log?**

---

## Part 3: The Deep Answer — Logarithm is FUNDAMENTAL

### 3.1 The Gradient Landscape Problem

Let's think about what happens during gradient descent **without** log, even with perfect precision.

**Setup**: Binary classification, predicting $P(y=1|x) = \sigma(w \cdot x)$

**Direct likelihood** (no log):
$$L = \prod_{i: y_i=1} p_i \cdot \prod_{j: y_j=0} (1-p_j)$$

where $p_i = \sigma(w \cdot x_i)$

**Gradient of direct likelihood**:
$$\frac{\partial L}{\partial w} = L \cdot \sum_i \left[ y_i \frac{1}{p_i} - (1-y_i)\frac{1}{1-p_i} \right] \cdot \frac{\partial p_i}{\partial w}$$

Notice: **The gradient is proportional to $L$ itself!**

#### Why This is Catastrophic

| Scenario | $L$ | Gradient magnitude | Problem |
|----------|-----|-------------------|---------|
| Good model | $L \approx 1$ | Normal | OK |
| Mediocre model | $L \approx 10^{-100}$ | Tiny | Vanishing gradients |
| Bad model | $L \approx 10^{-10000}$ | Essentially 0 | **No learning signal!** |

**The models that need to learn the most get the weakest gradients!**

This isn't a numerical precision issue — it's a **fundamental scaling problem**.

### 3.2 Log-Likelihood Fixes the Scaling

With log-likelihood:
$$\ell = \log L = \sum_i \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]$$

$$\frac{\partial \ell}{\partial w} = \sum_i \left[ y_i \frac{1}{p_i} - (1-y_i)\frac{1}{1-p_i} \right] \cdot \frac{\partial p_i}{\partial w}$$

**The gradient is NOT scaled by $L$!** Each sample contributes independently.

This is the **score function** in statistics — it has beautiful properties:
- Zero mean under the true distribution: $E[\nabla_\theta \log p(x|\theta)] = 0$
- Variance = Fisher Information (fundamental limit of learning)

### 3.3 The Deep Connection: Logarithm Creates Additive Structure

#### Independent Events Must Contribute Additively

Consider training on independent samples $x_1, x_2, ..., x_n$.

**Physical intuition**: If I learn from $x_1$, then separately learn from $x_2$, the total learning should be the sum, not the product.

**Mathematically**:
- Likelihoods multiply: $L = L_1 \cdot L_2$
- But "information" or "learning signal" should add: $I = I_1 + I_2$

The ONLY way to convert multiplicative to additive: **logarithm**.

$$\log L = \log L_1 + \log L_2$$

This isn't a convenience — it's **required** for independent contributions to learning.

### 3.4 Information-Theoretic Necessity

#### Shannon's Derivation

Shannon asked: "How should we measure information/surprise?"

Requirements:
1. More surprising events (lower probability) → more information
2. Certain events ($P = 1$) → zero information  
3. **Independent events: information adds**

From requirement 3:
$$I(P_1 \cdot P_2) = I(P_1) + I(P_2)$$

The ONLY function satisfying this: $I(P) = -\log P$

**Information = negative log probability**

This isn't a choice — it's **mathematically forced** by the additivity requirement.

### 3.5 Cross-Entropy Loss: Why That Specific Form?

The cross-entropy loss:
$$H(p, q) = -\sum_i p_i \log q_i$$

**Why log and not some other function?**

**Answer**: Cross-entropy measures the **expected number of bits** needed to encode samples from $p$ using a code optimized for $q$.

- If you use a code optimized for the wrong distribution, you waste bits
- The "waste" is exactly $D_{KL}(p || q)$ extra bits
- This only makes sense with log (bits are additive)

### 3.6 The Exponential Family Connection

Most distributions we use are "exponential family":
$$p(x|\theta) = h(x) \exp(\theta^T T(x) - A(\theta))$$

Examples: Gaussian, Bernoulli, Poisson, Categorical, etc.

**Key property**: Log-likelihood is LINEAR in parameters!
$$\log p(x|\theta) = \theta^T T(x) - A(\theta) + \log h(x)$$

This gives:
- Convex optimization (nice!)
- Sufficient statistics
- Conjugate priors

Without log, we lose this beautiful structure.

---

## Part 4: Thought Experiment — ML Without Logarithms

### Scenario: Perfect Computer

Imagine:
- Infinite precision arithmetic
- Can differentiate products exactly
- No underflow/overflow

### What Would Happen?

#### Problem 1: Gradient Magnitude Coupling

Training a model on 1 million samples:
$$L = \prod_{i=1}^{10^6} P(x_i|\theta)$$

Even if each $P(x_i|\theta) = 0.99$ (very confident):
$$L = 0.99^{10^6} \approx e^{-10000} \approx 10^{-4343}$$

Gradient: $\nabla L \propto L \approx 10^{-4343}$

Compare to a "worse" model with $P(x_i|\theta) = 0.9$:
$$L \approx 10^{-45000}, \quad \nabla L \approx 10^{-45000}$$

**The gradient ratio is $10^{-40657}$!**

Even with perfect precision, the optimizer sees:
- Good model: gradient = very small
- Bad model: gradient = incomprehensibly smaller

**Relative learning rates are broken.**

#### Problem 2: Loss of Locality

With log-likelihood, each sample's contribution is **local**:
$$\frac{\partial}{\partial \theta} \log P(x_i|\theta)$$

depends only on $x_i$ and current parameters.

Without log:
$$\frac{\partial}{\partial \theta} \prod_j P(x_j|\theta) = \sum_i \frac{\partial P(x_i|\theta)}{\partial \theta} \prod_{j \neq i} P(x_j|\theta)$$

Each term depends on **all other samples**. This breaks:
- Mini-batch training
- Stochastic gradient descent
- Distributed training

#### Problem 3: No Meaningful "Loss" Decomposition

We often want:
$$\text{Total Loss} = \sum_i \text{Loss}_i$$

This additivity enables:
- Per-sample analysis
- Curriculum learning
- Hard example mining
- Interpretability

Without log, we have:
$$\text{Total Likelihood} = \prod_i \text{Likelihood}_i$$

No easy way to ask "which samples contribute most to the loss?"

---

## Part 5: Could We Use a Different Function?

### What About $\sqrt{\cdot}$ or Other Transforms?

**Attempt**: Use $f(L) = L^{0.01}$ to compress the range.

**Problem**: 
$$f(L_1 \cdot L_2) = (L_1 \cdot L_2)^{0.01} \neq f(L_1) + f(L_2)$$

Doesn't convert products to sums. Gradients still couple.

### What About Normalizing Gradients?

**Attempt**: Always normalize $\nabla L$ to unit length.

**Problem**: Loses magnitude information!
- Can't distinguish "very wrong" from "slightly wrong"
- Adam/SGD momentum breaks
- Learning rate scheduling meaningless

### The Fundamental Truth

There's no escaping the mathematical structure:
- Independent probabilities multiply
- Learning signals must add
- Only logarithm bridges these

---

## Part 6: Summary — Why Logarithm is Fundamental

### Computational Reasons (Would Go Away with Perfect Computers)
- ❌ Numerical stability (underflow prevention)
- ❌ Faster computation (additions vs multiplications)

### Fundamental Reasons (Can Never Go Away)
- ✅ **Gradient scaling**: Without log, gradients scale with likelihood magnitude
- ✅ **Additivity**: Independent samples must contribute additively to learning
- ✅ **Information theory**: Surprise/information MUST be logarithmic (Shannon's theorem)
- ✅ **Locality**: Each sample's gradient contribution should be independent
- ✅ **Exponential family structure**: Log linearizes the most common distributions

### The Deep Insight

The logarithm isn't a hack to deal with small numbers. It's the **mathematical bridge** between:
- The **multiplicative** world of probabilities
- The **additive** world of learning/information

This bridge is **unique** (only log does this) and **necessary** (learning must be additive).

---

## Part 7: Implications for ML Practice

### Why Cross-Entropy Dominates

Cross-entropy loss:
$$L = -\sum_i y_i \log \hat{y}_i$$

This is log-likelihood for categorical distributions. It's not just "popular" — it's the **correct** loss for classification, because:
1. Measures actual information loss
2. Gradients scale properly
3. Each sample contributes independently

### Why MSE Works for Regression

For Gaussian noise:
$$p(y|x) \propto \exp\left(-\frac{(y - f(x))^2}{2\sigma^2}\right)$$

Log-likelihood:
$$\log p \propto -(y - f(x))^2$$

MSE **is** log-likelihood for Gaussian! That's why it works.

### The Softmax-CrossEntropy Pairing

$$\text{softmax}: z \rightarrow \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Log-softmax:
$$\log \text{softmax}(z)_i = z_i - \log \sum_j e^{z_j}$$

This cancels the exponential, giving **linear gradients** in logits!

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

Beautiful, simple, numerically stable — all because of log.

---

## Key Takeaways

1. **Logarithm is the UNIQUE function** converting products to sums
2. **Even with perfect computers**, direct likelihood optimization fails due to gradient scaling
3. **Information must be measured logarithmically** — Shannon proved this
4. **Learning signals must be additive** — log makes this possible
5. **Cross-entropy isn't arbitrary** — it's the mathematically correct loss

The logarithm isn't fighting against computer limitations. It's **aligned with the mathematical structure of probability and learning itself**.

---

## Exercises

1. **Gradient Scaling**: Compute $\nabla L$ and $\nabla \log L$ for $L = p_1 \cdot p_2 \cdot p_3$. Show that $\nabla \log L$ doesn't depend on $L$'s magnitude.

2. **Uniqueness Proof**: Prove that if $f(xy) = f(x) + f(y)$ and $f$ is continuous, then $f(x) = c \log x$ for some constant $c$.

3. **Information Additivity**: You flip two independent coins. Show that total information = sum of individual informations, which requires $I(p) = -\log p$.

4. **Alternative Loss**: Try training a simple model using $L = \prod p_i$ directly (with high-precision library). Observe the gradient behavior.

5. **Fisher Information**: Show that $\text{Var}[\nabla_\theta \log p(x|\theta)] = -E[\nabla^2_\theta \log p(x|\theta)]$ (only works with log!).
