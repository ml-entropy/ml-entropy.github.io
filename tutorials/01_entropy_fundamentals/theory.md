# Tutorial 1: Entropy Fundamentals & Information Theory

## 🎯 The Big Picture

**Machine Learning is compression.** When we train a model, we're finding a compact representation of the patterns in our data. The better the compression, the better the model understands the underlying structure.

---

## 1. What is Information?

### The Intuition: Surprise

Information is **surprise**. Events that are:
- **Rare** → High information (surprising when they happen)
- **Common** → Low information (expected, boring)

If someone tells you "the sun rose today," that's not informative—you expected it. But "there was an earthquake" is highly informative—it's surprising.

### Deriving the Information Formula

What mathematical function should measure "information" or "surprise"? Let's derive it from first principles!

**Requirements for a good information measure $I(p)$:**

1. **Monotonicity:** Less probable events should have more information.
   $$p_1 < p_2 \implies I(p_1) > I(p_2)$$

2. **Non-negativity:** Information should never be negative.
   $$I(p) \geq 0$$

3. **Certainty has no information:** If $P(x) = 1$, we already knew it would happen.
   $$I(1) = 0$$

4. **Additivity for independent events:** If $A$ and $B$ are independent, the information from observing both should add.
   $$I(P(A \cap B)) = I(P(A) \cdot P(B)) = I(P(A)) + I(P(B))$$

### The Key Step: What Function Satisfies Additivity?

The additivity requirement is the crucial constraint. We need:
$$I(p \cdot q) = I(p) + I(q)$$

**Question:** What function turns multiplication into addition?

**Answer:** The **logarithm**!

$$\log(p \cdot q) = \log(p) + \log(q)$$

So $I(p)$ must involve $\log(p)$. But $\log(p)$ is negative for $p < 1$, violating non-negativity.

**Solution:** Use $I(p) = -\log(p) = \log(1/p)$

### Verification

1. ✓ **Monotonicity:** If $p_1 < p_2$, then $-\log(p_1) > -\log(p_2)$
2. ✓ **Non-negativity:** For $0 < p \leq 1$, $-\log(p) \geq 0$
3. ✓ **Certainty:** $I(1) = -\log(1) = 0$
4. ✓ **Additivity:** $I(pq) = -\log(pq) = -\log(p) - \log(q) = I(p) + I(q)$

### Shannon's Formula

$$I(x) = -\log_2 P(x) = \log_2 \frac{1}{P(x)}$$

Where:
- $I(x)$ = information content of event $x$ (in bits)
- $P(x)$ = probability of event $x$
- Base 2 gives us "bits" (binary digits)

**Key properties (now derived, not just stated!):**
1. $I(x) \geq 0$ — Information is never negative
2. $I(x) = 0$ when $P(x) = 1$ — Certain events carry no information
3. $I(x) \to \infty$ as $P(x) \to 0$ — Impossible events would be infinitely surprising
4. $I(x,y) = I(x) + I(y)$ for independent events — Information adds

### Example: Coin Flips

| Event | Probability | Information |
|-------|-------------|-------------|
| Fair coin heads | 0.5 | $-\log_2(0.5) = 1$ bit |
| Biased coin (90% heads) | 0.9 | $-\log_2(0.9) ≈ 0.15$ bits |
| Biased coin (90% heads) tails | 0.1 | $-\log_2(0.1) ≈ 3.32$ bits |
| Roll a 6 on die | 1/6 | $-\log_2(1/6) ≈ 2.58$ bits |

---

## 2. Entropy: Expected Surprise (DISCRETE)

### Starting Simple: Discrete Random Variables

Before tackling continuous distributions, let's thoroughly understand the discrete case.

For a **discrete** random variable $X$ with PMF $p(x)$, entropy measures the **average information** we get from observing $X$.

### Derivation from First Principles

**Setup:** We observe $X$ many times. Sometimes $X = x_1$, sometimes $X = x_2$, etc.

**Question:** On average, how much information (surprise) do we get per observation?

**Step 1:** The information from observing $X = x$ is:
$$I(x) = -\log_2 p(x)$$

**Step 2:** The event $X = x$ occurs with probability $p(x)$.

**Step 3:** The **expected (average) information** is:
$$H(X) = \mathbb{E}[I(X)] = \sum_x p(x) \cdot I(x) = \sum_x p(x) \cdot (-\log_2 p(x))$$

### The Entropy Formula (Discrete)

$$\boxed{H(X) = -\sum_{x} p(x) \log_2 p(x)}$$

**This is just an expected value!** We're computing $\mathbb{E}[g(X)]$ where $g(x) = -\log_2 p(x)$.

### Why The Formula Makes Sense

Each term $-p(x) \log_2 p(x)$ contributes to entropy:
- **Common events:** High $p(x)$, low $-\log_2 p(x)$ → moderate contribution
- **Rare events:** Low $p(x)$, high $-\log_2 p(x)$, but weighted by low $p(x)$ → moderate contribution
- **Maximum contribution:** When $p(x)$ is "middling"

### Detailed Example: Fair Die

Let's compute entropy step by step for a fair 6-sided die.

| $x$ | $p(x)$ | $-\log_2 p(x)$ | $p(x) \cdot (-\log_2 p(x))$ |
|-----|--------|----------------|------------------------------|
| 1 | 1/6 | $\log_2 6 ≈ 2.585$ | $\frac{1}{6} \times 2.585 ≈ 0.431$ |
| 2 | 1/6 | $\log_2 6 ≈ 2.585$ | $\frac{1}{6} \times 2.585 ≈ 0.431$ |
| 3 | 1/6 | $\log_2 6 ≈ 2.585$ | $\frac{1}{6} \times 2.585 ≈ 0.431$ |
| 4 | 1/6 | $\log_2 6 ≈ 2.585$ | $\frac{1}{6} \times 2.585 ≈ 0.431$ |
| 5 | 1/6 | $\log_2 6 ≈ 2.585$ | $\frac{1}{6} \times 2.585 ≈ 0.431$ |
| 6 | 1/6 | $\log_2 6 ≈ 2.585$ | $\frac{1}{6} \times 2.585 ≈ 0.431$ |
| **Total** | 1 | | $H(X) ≈ 2.585$ bits |

So $H(\text{fair die}) = \log_2 6 ≈ 2.585$ bits.

### The Fair Coin vs Biased Coin

**Fair coin** ($p(H) = 0.5$):
$$H = -0.5 \log_2(0.5) - 0.5 \log_2(0.5) = -0.5 \times (-1) - 0.5 \times (-1) = 1 \text{ bit}$$

**Biased coin** ($p(H) = 0.9, p(T) = 0.1$):
$$H = -0.9 \log_2(0.9) - 0.1 \log_2(0.1)$$
$$= -0.9 \times (-0.152) - 0.1 \times (-3.322)$$
$$= 0.137 + 0.332 = 0.469 \text{ bits}$$

The biased coin is more predictable → lower entropy.

### Maximum Entropy for Discrete Distributions

**Theorem:** For a discrete distribution over $n$ outcomes, entropy is maximized when the distribution is **uniform**.

**Proof sketch:** Use Lagrange multipliers or Jensen's inequality. The intuition: any "peakedness" in the distribution makes outcomes more predictable, reducing entropy.

$$H_{max} = \log_2(n)$$

This is why we measure in bits—with $n = 2^k$ equally likely outcomes, you need exactly $k$ bits to specify which one occurred.

---

## 3. Entropy for Continuous Distributions (Differential Entropy)

### The Challenge

For continuous distributions, we can't just replace the sum with an integral. There's a subtlety!

### Attempt 1: Naive Substitution (WRONG)

If we naively write:
$$H(X) \stackrel{?}{=} -\int_{-\infty}^{\infty} f(x) \log f(x) \, dx$$

This is called **differential entropy**, and it has strange properties:
- Can be **negative**!
- Depends on units/scaling of $x$
- Not the "true" entropy of a continuous variable

### The Right Way: Start from Discrete

**Step 1:** Discretize $X$ into bins of width $\Delta x$.

Let $x_i$ be the center of bin $i$. The probability of being in bin $i$ is approximately:
$$p_i = P(x_i - \frac{\Delta x}{2} \leq X \leq x_i + \frac{\Delta x}{2}) \approx f(x_i) \Delta x$$

**Step 2:** Compute discrete entropy of the discretized distribution:
$$H_\Delta = -\sum_i p_i \log_2 p_i = -\sum_i f(x_i) \Delta x \cdot \log_2(f(x_i) \Delta x)$$

**Step 3:** Expand the logarithm:
$$H_\Delta = -\sum_i f(x_i) \Delta x \cdot [\log_2 f(x_i) + \log_2 \Delta x]$$
$$= -\sum_i f(x_i) \log_2 f(x_i) \cdot \Delta x - \log_2 \Delta x \sum_i f(x_i) \Delta x$$

**Step 4:** As $\Delta x \to 0$:
- First term → $-\int f(x) \log_2 f(x) \, dx = h(X)$ (differential entropy)
- Second term → $-\log_2 \Delta x \times 1 \to +\infty$

### The Key Insight

The "true" entropy of a continuous distribution is **infinite**! This makes sense: specifying a real number to infinite precision requires infinite bits.

But the **differential entropy** $h(X) = -\int f(x) \log f(x) \, dx$ is still useful because:
- The infinite part ($-\log_2 \Delta x$) is the same for all continuous distributions
- When comparing two distributions, this infinite part **cancels out**!

### Differential Entropy Formula

$$h(X) = -\int_{-\infty}^{\infty} f(x) \log f(x) \, dx$$

**Properties:**
- Can be negative (e.g., Uniform(0, 0.5) has $h = -1$ bit)
- Is NOT bounded below
- Depends on the scale of $x$
- BUT: Differences and KL divergences are well-defined

### Example: Gaussian Distribution

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$h(X) = -\int f(x) \log f(x) \, dx$$

where $f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

**Derivation:**
$$\log f(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

$$h(X) = -\int f(x) \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}\right] dx$$

$$= \frac{1}{2}\log(2\pi\sigma^2) \underbrace{\int f(x) dx}_{=1} + \frac{1}{2\sigma^2} \underbrace{\int (x-\mu)^2 f(x) dx}_{=\sigma^2}$$

$$= \frac{1}{2}\log(2\pi\sigma^2) + \frac{1}{2}$$

$$\boxed{h(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2}\log(2\pi e \sigma^2)}$$

Only depends on $\sigma$, not $\mu$! (Location doesn't affect uncertainty.)

---

## 3. Entropy as Optimal Code Length

### The Coding Perspective

Here's the profound insight: **entropy equals the minimum average code length**.

If you want to send messages from a distribution $P$, the best possible encoding uses (on average):

$$\text{Average code length} \geq H(P)$$

This is Shannon's **source coding theorem**.

### Why This Matters for ML

When we train a model with log-likelihood:
$$\mathcal{L} = -\sum_i \log P_\theta(x_i)$$

We're literally asking: *"How many bits does our model need to encode the training data?"*

A better model = fewer bits = lower loss = better compression.

---

## 4. Huffman Coding: Achieving the Entropy Bound

### The Algorithm

Huffman coding constructs an optimal prefix-free code:

1. Create a leaf node for each symbol with its probability
2. Repeat until one node remains:
   - Remove the two nodes with lowest probability
   - Create a new internal node with these as children
   - Assign probability = sum of children's probabilities
3. Assign 0 to left branches, 1 to right branches
4. Code for each symbol = path from root to leaf

### Example: Weather Forecasts

Suppose weather probabilities are:
- Sunny: 0.5
- Cloudy: 0.25
- Rainy: 0.125
- Snowy: 0.125

**Huffman tree:**
```
        (1.0)
       /     \
    (0.5)   (0.5)
    Sunny   /    \
        (0.25)  (0.25)
        Cloudy  /    \
            (0.125) (0.125)
            Rainy   Snowy
```

**Codes:**
- Sunny: `0` (1 bit)
- Cloudy: `10` (2 bits)
- Rainy: `110` (3 bits)
- Snowy: `111` (3 bits)

**Average code length:**
$$0.5 \times 1 + 0.25 \times 2 + 0.125 \times 3 + 0.125 \times 3 = 1.75 \text{ bits}$$

**Entropy:**
$$H = -0.5\log_2(0.5) - 0.25\log_2(0.25) - 2 \times 0.125\log_2(0.125) = 1.75 \text{ bits}$$

The Huffman code achieves the entropy bound exactly (when probabilities are powers of 2).

---

## 5. Cross-Entropy: Using the Wrong Code

### The Setup

What if we don't know the true distribution $P$ and instead use our model distribution $Q$ to encode data?

**Cross-entropy** measures the average code length when using $Q$ to encode samples from $P$:

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

Or continuously:
$$H(P, Q) = -\int p(x) \log q(x) \, dx$$

### Key Insight

$$H(P, Q) \geq H(P)$$

Using the wrong distribution always costs more bits (or the same if $Q = P$).

The "extra bits" is exactly the **KL divergence** (next tutorial):
$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

### Cross-Entropy Loss in ML

When we train a classifier, we minimize:
$$\mathcal{L} = -\sum_i \log Q_\theta(y_i | x_i)$$

This is cross-entropy! We're:
1. Using true labels $y_i$ (samples from $P$)
2. Asking how many bits our model $Q_\theta$ needs
3. Minimizing this → making $Q$ approach $P$

---

## 6. ML Through the Entropy Lens

### Supervised Learning

**Goal:** Learn $P(Y|X)$ from data

**Loss:** Cross-entropy = $-\log P_\theta(y|x)$

**What we're doing:** Finding model parameters that compress the labels most efficiently given the inputs.

### Unsupervised Learning

**Goal:** Learn $P(X)$

**Loss:** Negative log-likelihood = $-\log P_\theta(x)$

**What we're doing:** Building a compression algorithm for the data. Better model = shorter description length.

### The Minimum Description Length Principle

The best model is the one that provides the shortest total description:

$$\text{Total length} = \underbrace{L(\text{model})}_{\text{complexity}} + \underbrace{L(\text{data}|\text{model})}_{\text{fit}}$$

This is the formal version of Occam's Razor!

- Too simple model: Short $L(\text{model})$, long $L(\text{data}|\text{model})$
- Too complex model: Long $L(\text{model})$, short $L(\text{data}|\text{model})$
- Just right: Minimum total length

### Regularization = Model Code Length

When we add L2 regularization:
$$\mathcal{L} = -\log P_\theta(y|x) + \lambda \|\theta\|^2$$

The $\lambda \|\theta\|^2$ term is like a "description length" for the parameters. We're saying: "Prefer models that can be described with smaller numbers."

---

## 7. Summary: The Entropy Worldview

| Concept | Entropy Interpretation |
|---------|----------------------|
| Information | Surprise / unexpectedness |
| Entropy | Expected surprise / uncertainty |
| Low entropy | Predictable, structured, compressible |
| High entropy | Random, unstructured, incompressible |
| Learning | Finding patterns that reduce entropy |
| Overfitting | Memorizing noise (incompressible) |
| Generalization | Capturing true low-entropy structure |
| Model capacity | Bits available for storing patterns |

### The Deep Insight

**The universe is not random—it has structure.** 

Data from the real world is not maximum entropy; it has patterns, regularities, and dependencies. Machine learning is the art of discovering and encoding this structure.

A model that assigns high probability to real data (low cross-entropy) has discovered genuine structure. A model that memorizes training data has wasted its bits on noise.

---

## Next: KL Divergence

Now that we understand entropy and cross-entropy, we're ready to explore **KL divergence**—the fundamental measure of how different two distributions are.
